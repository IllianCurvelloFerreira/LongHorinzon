"""Auditoria reproduzível de dados, previsões e sMAPE do GBT.

Execute a partir da raiz do repositório LongHorinzon:
  python -m scripts.diagnose_gbt --mode audit --data Weather
  python -m scripts.diagnose_gbt --mode train --data Weather --horizon 96 --train_epochs 10
  python -m scripts.diagnose_gbt --mode all --data Weather --horizon 96 --train_epochs 10

Não modifica arquivos existentes do projeto nem usa o teste para selecionar modelos.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader.preprocess_ett import load_ett_long
from datasets.ett_gbt import ETTGBTDataset, ensure_ett_csv, ensure_ett_multivariate_csv
from models.gbt.model import GBTVanillaStandalone
from training.engine_gbt import (
    build_loaders, parse_s_layers, process_batch, train_one_stage, resolve_target_idx,
)
from utils.device import get_device
from utils.seed import set_seed


def args_parser():
    p = argparse.ArgumentParser(description='Diagnóstico independente do GBT')
    p.add_argument('--mode', choices=['audit', 'train', 'all'], default='all')
    p.add_argument('--data', choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange'], default='Weather')
    p.add_argument('--root_path', default='./data/ETT')
    p.add_argument('--data_dir', default='./nixtla_cache')
    p.add_argument('--output_dir', default='./results/diagnostics')
    p.add_argument('--target', default='OT')
    p.add_argument('--horizon', type=int, default=96)
    p.add_argument('--seq_len', type=int, default=None)
    p.add_argument('--label_len', type=int, default=None)
    p.add_argument('--pred_len', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train_epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.05)
    p.add_argument('--fd_model', type=int, default=32)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--d_layers', type=int, default=2)
    p.add_argument('--s_layers', default='3,2,1')
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--criterion', choices=['Standard', 'MaxAbs'], default='Standard')
    p.add_argument('--split_mode', choices=['legacy', 'ratio'], default='ratio')
    p.add_argument('--train_ratio', type=float, default=0.7)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--patience', type=int, default=3)  # mantido por compatibilidade com o engine
    p.add_argument('--time', action='store_true')
    p.add_argument('--test_inverse', action='store_true')
    p.add_argument('--features', default='S')
    p.add_argument('--max_saved_points', type=int, default=250000)
    p.add_argument('--skip_raw_source', action='store_true', help='Não baixa/inspeciona a fonte long para auditoria')
    a = p.parse_args()
    a.seq_len = a.horizon if a.seq_len is None else a.seq_len
    a.label_len = a.seq_len if a.label_len is None else a.label_len
    a.pred_len = a.horizon if a.pred_len is None else a.pred_len
    if a.label_len > a.seq_len:
        p.error('--label_len não pode exceder --seq_len')
    return a


def summary(arr):
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if not a.size:
        return {'n': 0}
    return {
        'n': int(a.size), 'mean': float(a.mean()), 'std': float(a.std()),
        'min': float(a.min()), 'p01': float(np.percentile(a, 1)),
        'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)),
        'p99': float(np.percentile(a, 99)), 'max': float(a.max()),
    }


def audit_data(a, out):
    print('\n' + '=' * 75 + '\nAUDITORIA DOS CSVs / FONTE\n' + '=' * 75)
    paths = {
        'univariate': ensure_ett_csv(a.root_path, a.data, a.target, a.data_dir),
        'multivariate': ensure_ett_multivariate_csv(a.root_path, a.data, a.target, a.data_dir),
    }
    data = {}
    report = {'dataset': a.data, 'files': {}}
    for mode, path in paths.items():
        df = pd.read_csv(path)
        data[mode] = df
        cols = [col for col in df if col != 'date']
        info = {
            'path': str(path), 'rows': len(df), 'columns': cols,
            'n_variables': len(cols), 'duplicate_dates': int(df.date.duplicated().sum()),
            'missing_cells': int(df.isna().sum().sum()), 'target_stats': summary(df[a.target]),
            'target_near_zero_pct': float(100 * (df[a.target].abs() < .01).mean()),
            'target_nonpositive_pct': float(100 * (df[a.target] <= 0).mean()),
        }
        report['files'][mode] = info
        print(f'\n{mode}: {info}')
        if mode == 'multivariate' and len(cols) <= 1:
            print('ALERTA: arquivo multivariado contém apenas 1 variável!')
    u = data['univariate'][['date', a.target]]
    m = data['multivariate'][['date', a.target]]
    merged = u.merge(m, on='date', how='outer', suffixes=('_uni', '_multi'), indicator=True, validate='one_to_one')
    matching = merged[merged._merge == 'both'].copy()
    diff = (matching[a.target + '_uni'] - matching[a.target + '_multi']).abs()
    comparison = {
        'matched_dates': len(matching),
        'left_only_dates': int((merged._merge == 'left_only').sum()),
        'right_only_dates': int((merged._merge == 'right_only').sum()),
        'target_max_abs_difference': float(diff.max()) if len(diff) else None,
        'target_mean_abs_difference': float(diff.mean()) if len(diff) else None,
        'target_different_points': int((diff > 1e-6).sum()),
    }
    report['target_comparison'] = comparison
    print('\nCOMPARAÇÃO ENTRE OS DOIS TARGETS:', comparison)
    matching['abs_diff'] = diff
    matching.nlargest(25, 'abs_diff').to_csv(out / f'{a.data}_target_differences.csv', index=False)

    # Audita dados raw antes da interpolação (quando disponíveis).
    if not a.skip_raw_source:
        try:
            raw = load_ett_long(data_dir=a.data_dir, group=a.data, normalize=False)
            report['raw_source'] = {
                'rows': len(raw), 'columns': list(raw.columns),
                'unique_ids': sorted(map(str, raw.unique_id.unique())) if 'unique_id' in raw else [],
                'missing_y': int(raw.y.isna().sum()) if 'y' in raw else None,
                'sentinel_minus9999': int((raw.y == -9999).sum()) if 'y' in raw else None,
            }
            print('\nFONTE ORIGINAL:', report['raw_source'])
        except Exception as e:
            report['raw_source_error'] = repr(e)
            print('\n[AVISO] Não foi possível inspecionar a fonte original:', e)

    with (out / f'{a.data}_data_audit.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def point_stats(y, p):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y) & np.isfinite(p)
    if not np.all(finite):
        raise ValueError(f'Previsões ou targets não finitos: {(~finite).sum()} pontos')
    ae = np.abs(p - y)
    den = np.abs(y) + np.abs(p)
    smape = 200 * ae / np.maximum(den, 1e-8)
    bins = [(0, .01), (.01, .1), (.1, 1), (1, 2), (2, 5), (5, 10), (10, np.inf)]
    by_bin = []
    for lo, hi in bins:
        select = (np.abs(y) >= lo) & (np.abs(y) < hi)
        by_bin.append({
            'bin': f'[{lo}, {hi})', 'n': int(select.sum()),
            'share_pct': float(100 * select.mean()),
            'smape_pct': float(smape[select].mean()) if select.any() else None,
            'mae_original': float(ae[select].mean()) if select.any() else None,
            'contribution_to_total_smape_pp': float(smape[select].sum() / len(y)),
        })
    corr = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else None
    return {
        'mse_original': float(np.mean((p - y) ** 2)),
        'mae_original': float(ae.mean()),
        'smape_original_pct': float(smape.mean()),
        'signed_opposition_pct': float(100 * np.mean(y * p < 0)),
        'smape_ge100_pct': float(100 * np.mean(smape >= 100)),
        'smape_ge190_pct': float(100 * np.mean(smape >= 190)),
        'corr': corr, 'y_true': summary(y), 'y_pred': summary(p),
        'abs_error': summary(ae), 'smape_point': summary(smape), 'bins': by_bin,
    }


@torch.no_grad()
def predict_stage(model, loader, dataset, a, device, stage):
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        pred, true = process_batch(model, batch, a.label_len, a.pred_len, device, stage)
        predictions.append(pred.detach().cpu().numpy())
        targets.append(true.detach().cpu().numpy())
    p = np.concatenate(predictions, axis=0)
    y = np.concatenate(targets, axis=0)
    return (p, y, dataset.inverse_transform_y(p), dataset.inverse_transform_y(y))


def save_points(a, out, mode, stage, raw_pred, raw_true, org_pred, org_true):
    # Salva coordenadas (origem, passo), permitindo diagnosticar a posição do horizonte.
    count, horizon, channels = org_pred.shape
    if channels != 1:
        raise ValueError('Diagnóstico espera previsão de alvo único (c_out=1)')
    if count * horizon > a.max_saved_points:
        print(f'[AVISO] CSV de pontos amostrado: {count*horizon} > {a.max_saved_points}')
        keep = max(1, a.max_saved_points // horizon)
    else:
        keep = count
    origin_idxs = np.arange(keep)
    pred = org_pred[:keep, :, 0]
    true = org_true[:keep, :, 0]
    smape = 200 * np.abs(pred - true) / np.maximum(np.abs(pred) + np.abs(true), 1e-8)
    path = out / f'{a.data}_H{a.pred_len}_{mode}_{stage}_points.csv'
    df = pd.DataFrame({
        'window': np.repeat(origin_idxs, horizon),
        'step_ahead': np.tile(np.arange(1, horizon + 1), keep),
        'y_true_original': true.reshape(-1),
        'y_pred_original': pred.reshape(-1),
        'abs_error_original': np.abs(pred - true).reshape(-1),
        'smape_pct': smape.reshape(-1),
        'y_true_standardized': raw_true[:keep].reshape(-1),
        'y_pred_standardized': raw_pred[:keep].reshape(-1),
    })
    df.to_csv(path, index=False)
    print('Pontos exportados:', path)


def experiment(a, mode, out, device):
    set_seed(a.seed)
    a.input_mode = mode
    print('\n' + '='*75 + f'\nTREINAMENTO {a.data} H={a.pred_len} | {mode} | seed={a.seed}\n' + '='*75)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(a)
    if mode == 'multivariate' and train_ds.enc_in <= 1:
        raise ValueError(f'{a.data} multivariado possui enc_in={train_ds.enc_in}. Corrija o dataset.')
    model = GBTVanillaStandalone(
        enc_in=train_ds.enc_in, dec_in=1, c_out=1,
        label_len=a.label_len, pred_len=a.pred_len,
        fd_model=a.fd_model, d_model=a.d_model, d_layers=a.d_layers,
        n_heads=a.n_heads, s_layers=parse_s_layers(a.s_layers),
        dropout=a.dropout, use_time=a.time,
        with_minute=(a.data in {'ETTm1', 'ETTm2'}),
        target_idx=resolve_target_idx(train_ds, a), input_mode=mode,
    ).to(device)
    first_state = train_one_stage(model, train_loader, val_loader, val_ds, a, device, 'first stage')
    model.load_state_dict(first_state)
    # Preserva cópia da Stage 1 para diagnóstico, sem seleção pelo teste.
    first_model = copy.deepcopy(model)
    second_state = train_one_stage(model, train_loader, val_loader, val_ds, a, device, 'second stage')
    model.load_state_dict(second_state)
    result = {'dataset': a.data, 'mode': mode, 'seed': a.seed,
              'horizon': a.pred_len, 'n_input_variables': train_ds.enc_in,
              'columns': train_ds.x_cols, 'stages': {}}
    for stage, m in [('stage1', first_model), ('stage2', model)]:
        engine_stage = 'first stage' if stage == 'stage1' else 'second stage'
        p, y, p_orig, y_orig = predict_stage(m, test_loader, test_ds, a, device, engine_stage)
        details = point_stats(y_orig, p_orig)
        details['mse_standardized'] = float(np.mean((p - y) ** 2))
        details['mae_standardized'] = float(np.mean(np.abs(p - y)))
        result['stages'][stage] = details
        print(f'\n{mode} {stage}: MSE normalizado={details["mse_standardized"]:.6f} '
              f'MAE normalizado={details["mae_standardized"]:.6f} '
              f'sMAPE original={details["smape_original_pct"]:.2f}%')
        print('Correlação:', details['corr'], '| Sinais opostos:',
              f'{details["signed_opposition_pct"]:.2f}%',
              '| sMAPE >= 190%:', f'{details["smape_ge190_pct"]:.2f}%')
        print(pd.DataFrame(details['bins']).to_string(index=False))
        save_points(a, out, mode, stage, p, y, p_orig, y_orig)
    return result


def main():
    a = args_parser()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {'config': vars(a).copy()}
    if a.mode in {'audit', 'all'}:
        report['audit'] = audit_data(a, out)
    if a.mode in {'train', 'all'}:
        device = get_device()
        report['experiments'] = {}
        for mode in ['univariate', 'multivariate']:
            result = experiment(a, mode, out, device)
            report['experiments'][mode] = result
        u = report['experiments']['univariate']['stages']
        m = report['experiments']['multivariate']['stages']
        print('\n' + '='*75 + '\nCOMPARAÇÃO FINAL NA MESMA SEED\n' + '='*75)
        rows = []
        for stage in ['stage1', 'stage2']:
            for metric in ['mse_standardized', 'mae_standardized', 'smape_original_pct']:
                rows.append({'stage': stage, 'metric': metric,
                             'univariate': u[stage][metric], 'multivariate': m[stage][metric],
                             'delta_multi_minus_uni': m[stage][metric]-u[stage][metric]})
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        df.to_csv(out / f'{a.data}_H{a.pred_len}_comparison.csv', index=False)
    with (out / f'{a.data}_H{a.pred_len}_full_diagnostic.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('\nArquivos de diagnóstico salvos em:', out)


if __name__ == '__main__':
    main()

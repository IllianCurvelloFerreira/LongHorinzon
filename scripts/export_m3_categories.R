# Exporta os metadados OFICIAIS da M3 mensal, com categoria economica.
# Requer R e pacote Mcomp. Nao executa previsoes e nao modifica os dados de treino.
# Execucao na raiz do repositorio: Rscript scripts/export_m3_categories.R

if (!requireNamespace("Mcomp", quietly = TRUE)) {
  install.packages("Mcomp", repos = "https://cloud.r-project.org")
}

data("M3", package = "Mcomp")

rows <- lapply(M3, function(s) {
  data.frame(
    series_id = as.character(s$sn),
    frequency = toupper(as.character(s$period)),
    category = toupper(as.character(s$type)),
    description = if (length(s$description)) as.character(s$description)[1] else NA_character_,
    stringsAsFactors = FALSE
  )
})

metadata <- do.call(rbind, rows)
monthly <- metadata[tolower(metadata$frequency) == "monthly", ]
monthly <- monthly[order(as.integer(sub("^[^0-9]+", "", monthly$series_id))), ]

expected <- c(DEMOGRAPHIC=111L, FINANCE=145L, INDUSTRY=334L,
              MACRO=312L, MICRO=474L, OTHER=52L)
actual <- table(factor(monthly$category, levels=names(expected)))
if (nrow(monthly) != 1428L || !identical(as.integer(actual), as.integer(expected))) {
  stop("Metadados inesperados da M3 mensal; confira versao/fonte antes de usar. ",
       "Total: ", nrow(monthly), "; Categorias: ",
       paste(names(actual), actual, sep="=", collapse=", "))
}
if (anyDuplicated(monthly$series_id)) {
  stop("series_id duplicado no pacote Mcomp")
}

out <- "data/M3/m3_monthly_categories.csv"
dir.create(dirname(out), recursive=TRUE, showWarnings=FALSE)
write.csv(monthly, out, row.names=FALSE, na="")
cat("Arquivo gerado:", out, "\n")
print(table(monthly$category))
cat("Exemplos:\n")
print(utils::head(monthly, 4))

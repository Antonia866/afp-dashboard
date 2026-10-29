# Datos automáticos para los monitores de Bancos y Retail

La tarea de GitHub `.github/workflows/datos_monitores.yml` corre todos los días y descarga:

- **CMF** – Reporte Mensual de Información Financiera del Sistema Bancario (un Excel por mes) → `cmf/reporte_mensual_AAAA-MM.xlsx`.
- **INE** – cuadros estadísticos de Actividad Mensual del Comercio (IAC, IACM, ISUP) → `ine/`.

`manifest.json` registra qué se descargó y cuándo. La actualización diaria de los monitores lee estos archivos desde este repositorio público y valida los meses anteriores antes de escribir datos nuevos. No hay que hacer nada a mano.

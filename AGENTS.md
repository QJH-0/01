# Project Rules

## Encoding

1. All text files in this repository must use UTF-8 without BOM.
2. Before editing any existing text file, detect its current encoding first. If the file is not UTF-8, or the encoding cannot be confirmed safely, do not overwrite it directly.
3. When a legacy-encoded file must be changed, convert it to UTF-8 without BOM in a controlled step first, then apply the content edit.
4. For PowerShell-based file writes, use UTF-8 without BOM explicitly, for example `[System.Text.UTF8Encoding]::new($false)`, and do not rely on shell defaults.
5. If terminal output looks garbled, treat that as an encoding warning and stop editing the file until the encoding is verified.

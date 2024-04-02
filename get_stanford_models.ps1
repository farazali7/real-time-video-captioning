# This PowerShell script downloads the Stanford CoreNLP models.

$CORENLP = "stanford-corenlp-full-2015-12-09"
$SPICELIB = "pycocoevalcap/spice/lib"
$JAR = "stanford-corenlp-3.6.0"

$DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
cd $DIR

if (Test-Path "$SPICELIB/$JAR.jar") {
    Write-Host "Found Stanford CoreNLP."
} else {
    Write-Host "Downloading..."
    Invoke-WebRequest -Uri "http://nlp.stanford.edu/software/$CORENLP.zip" -OutFile "$CORENLP.zip"
    Write-Host "Unzipping..."
    Expand-Archive -LiteralPath "$CORENLP.zip" -DestinationPath "$SPICELIB/"
    Move-Item -Path "$SPICELIB/$CORENLP/$JAR.jar" -Destination "$SPICELIB/"
    Move-Item -Path "$SPICELIB/$CORENLP/$JAR-models.jar" -Destination "$SPICELIB/"
    Remove-Item -Path "$CORENLP.zip"
    Remove-Item -Path "$SPICELIB/$CORENLP" -Recurse
    Write-Host "Done."
}

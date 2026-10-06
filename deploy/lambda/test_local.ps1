# Test the Lambda image locally with the Lambda Runtime Interface Emulator bundled in the base image.
# Does not touch AWS. Requires Docker Desktop. Simulates Lambda's ceiling: 10 GB of RAM and 6 vCPUs.
# A desktop CPU is not Lambda's CPU, so latencies are only indicative; peak memory is still useful
# to check whether the model fits in 10,240 MB.
#
# Usage (from deploy\lambda):  powershell -ExecutionPolicy Bypass -File test_local.ps1 [-KeepBf16] [-N 20]
param([switch]$KeepBf16, [int]$N = 20)

$ErrorActionPreference = "Stop"
$Image = "strands-decider-lambda:x86"
$ResultsDir = Join-Path $PSScriptRoot "..\..\results\local_lambda"
New-Item -ItemType Directory -Force $ResultsDir | Out-Null
$Variant = if ($KeepBf16) { "bf16" } else { "fp32" }
$OutFile = Join-Path $ResultsDir "lambda_local_$Variant.csv"
$PeakFile = Join-Path $env:TEMP "decider_peak.txt"

docker buildx build --platform linux/amd64 -t $Image --load $PSScriptRoot
$KeepBf16Value = if ($KeepBf16) { "1" } else { "0" }
docker rm -f decider-lambda 2>$null | Out-Null
docker run -d --name decider-lambda -p 9000:8080 --memory 10g --cpus 6 -e DECIDER_KEEP_BF16=$KeepBf16Value $Image | Out-Null
Start-Sleep -Seconds 3

$Url = "http://127.0.0.1:9000/2015-03-31/functions/function/invocations"
$Body = '{"state": "Help! My payouts have been failing for 3 days!", "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "", "sales": "", "retail": ""}}, "urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}'

$Job = Start-Job -ArgumentList $PeakFile -ScriptBlock {
    param($PeakFile)
    $Max = 0.0
    while ($true) {
        $Usage = docker stats decider-lambda --no-stream --format "{{.MemUsage}}"
        if ($Usage -match "([\d\.]+)GiB") { $Value = [double]$Matches[1]; if ($Value -gt $Max) { $Max = $Value; Set-Content $PeakFile $Max } }
        Start-Sleep -Milliseconds 500
    }
}

"i,cold,client_ms,latency_ms,cold_start_load_ms,choice,confidence,urgent" | Set-Content $OutFile -Encoding utf8
for ($i = 0; $i -le $N; $i++) {
    $Watch = [Diagnostics.Stopwatch]::StartNew()
    $Response = Invoke-RestMethod -Uri $Url -Method Post -Body $Body -ContentType "application/json" -TimeoutSec 900
    $Watch.Stop()
    $ClientMs = [math]::Round($Watch.Elapsed.TotalMilliseconds, 1)
    "$i,$($i -eq 0),$ClientMs,$($Response.latency_ms),$($Response.cold_start_load_ms),$($Response.answers.team.choice),$($Response.answers.team.confidence),$($Response.answers.urgent.noul)" | Add-Content $OutFile
    Write-Host "#$i client=$ClientMs ms  latency_ms=$($Response.latency_ms)  load=$($Response.cold_start_load_ms)  choice=$($Response.answers.team.choice) ($($Response.answers.team.confidence))"
}
Stop-Job $Job; Remove-Job $Job
$Peak = Get-Content $PeakFile -ErrorAction SilentlyContinue
Write-Host "Container peak memory: $Peak GiB (limit 10 GiB)"
Add-Content $OutFile "# peak_memory_gib=$Peak keep_bf16=$KeepBf16Value"
docker rm -f decider-lambda | Out-Null

$ErrorActionPreference = "Continue"
$python = "D:\anaconda\envs\smd-blackwell\python.exe"
$root = "D:\Desktop\my_project\quadrotor_diffusion_ppo"
$runs = @(
    @("pure_ppo", "20260812"),
    @("pure_ppo", "20260813"),
    @("pure_ppo", "20260814"),
    @("diffusion_ppo", "20260812"),
    @("diffusion_ppo", "20260813"),
    @("diffusion_ppo", "20260814")
)

Set-Location -LiteralPath $root
foreach ($run in $runs) {
    $method = $run[0]
    $seed = $run[1]
    $stdout = Join-Path $root "logs\s6_${method}_seed_${seed}.stdout.log"
    $stderr = Join-Path $root "logs\s6_${method}_seed_${seed}.stderr.log"
    & $python "scripts/run_s6_core.py" --method $method --seed $seed --phase all 1>> $stdout 2>> $stderr
    if ($LASTEXITCODE -ne 0) {
        throw "S6 run failed: method=$method seed=$seed exit=$LASTEXITCODE"
    }
}
& $python "scripts/analyze_s6_core.py" 1>> "logs\s6_analysis.stdout.log" 2>> "logs\s6_analysis.stderr.log"
if ($LASTEXITCODE -ne 0) {
    throw "S6 analysis failed: exit=$LASTEXITCODE"
}

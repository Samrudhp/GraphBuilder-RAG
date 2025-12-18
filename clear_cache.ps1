# Clear all Python cache files
Write-Host "Clearing all __pycache__ directories..." -ForegroundColor Yellow

# Remove all __pycache__ directories
Get-ChildItem -Path . -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force

# Remove all .pyc files
Get-ChildItem -Path . -Recurse -Filter "*.pyc" -File | Remove-Item -Force

# Remove all .pyo files
Get-ChildItem -Path . -Recurse -Filter "*.pyo" -File | Remove-Item -Force

Write-Host "✅ Cache cleared successfully!" -ForegroundColor Green

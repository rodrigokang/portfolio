$body = @{
    text = "The product arrived on time and works exactly as expected."
} | ConvertTo-Json

Invoke-RestMethod `
    -Method POST `
    -Uri "http://localhost:7071/api/predict" `
    -ContentType "application/json" `
    -Body $body

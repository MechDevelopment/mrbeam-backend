curl -X 'POST' \
  'http://localhost:8001/api/v1/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test.jpg;type=image/jpeg'

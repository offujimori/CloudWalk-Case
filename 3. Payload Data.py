import requests


## Send the data to the address url after it running it
## Retrieve answer
url = "http://127.0.0.1:5000/fraud_check"
payload = {
    "transaction_id": 2342357,
    "merchant_id": 29744,
    "user_id": 97051,
    "card_number": "434505******9116",
    "transaction_date": "2019-11-30T23:16:32.812632",
    "transaction_amount": 373,
    "device_id": 285475
}

headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print("Response Code:", response.status_code)
print("Response Content:", response.content)

input()
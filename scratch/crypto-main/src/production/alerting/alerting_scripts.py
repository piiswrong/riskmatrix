import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode
import datetime

class BinanceFutures:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://fapi.binance.com'

    def _generate_signature(self, params):
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _get_timestamp(self):
        return int(time.time() * 1000)

    def get_account_info(self):
        endpoint = '/fapi/v3/account'
        params = {
            'timestamp': self._get_timestamp()
        }
        params['signature'] = self._generate_signature(params)
        
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        response = requests.get(
            self.base_url + endpoint,
            params=params,
            headers=headers
        )
        return response.json()

    def get_balance(self):
        endpoint = '/fapi/v3/balance'
        params = {
            'timestamp': self._get_timestamp()
        }
        params['signature'] = self._generate_signature(params)
        
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        response = requests.get(
            self.base_url + endpoint,
            params=params,
            headers=headers
        )
        return response.json()

    def get_position_risk(self):
        endpoint = '/fapi/v3/positionRisk'
        params = {
            'timestamp': self._get_timestamp()
        }
        params['signature'] = self._generate_signature(params)
        
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        response = requests.get(
            self.base_url + endpoint,
            params=params,
            headers=headers
        )
        return response.json()

class FeishuBot:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.headers = {
            "Content-Type": "application/json",
            "charset": "utf-8"
        }

    def send_text(self, text):
        message = {
            "msg_type": "text",
            "content": {
                "text": text
            }
        }
        
        response = requests.post(
            url=self.webhook_url,
            headers=self.headers,
            json=message
        )
        return response.json()

    def send_rich_text(self, title, content):
        message = {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": title,
                        "content": content
                    }
                }
            }
        }
        
        response = requests.post(
            url=self.webhook_url,
            headers=self.headers,
            json=message
        )
        return response.json()

api_key = 'bXR89e8bANMZseJhLpj867dgNn6y53SrsSmApZPyFHAxst2BEeDKDztd8ogU1t2X'
api_secret = 'rQXoVq8Aa7S4ssjEily8BgEH98YgyMNX4bEO70dIukYyd4pHgzwPWHxAwnpKpFGg'

client = BinanceFutures(api_key, api_secret)

# Get account information including margins
account_info = client.get_account_info()
# Get position risk including unrealized PnL
positions = client.get_position_risk()
total_unrealized_pnl = sum(float(position.get('unRealizedProfit')) for position in positions)

message_to_be_sent = "Account Information:\n"
# print out date as YYYY-MM-DD
message_to_be_sent += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n"
message_to_be_sent += f"Total Initial Margin: {account_info.get('totalInitialMargin')}\n"
message_to_be_sent += f"Total Maintenance Margin: {account_info.get('totalMaintMargin')}\n" 
message_to_be_sent += f"Total Margin Balance: {account_info.get('totalMarginBalance')}\n"

message_to_be_sent += "\nPosition Risk Information:\n"
message_to_be_sent += f"total_unrealized_pnl: {total_unrealized_pnl}\n"
for position in positions:
    message_to_be_sent += f"Symbol: {position.get('symbol')}\t"
    message_to_be_sent += f"Unrealized PnL: {position.get('unRealizedProfit')}\t"
    message_to_be_sent += f"Leverage: {position.get('leverage')}\n"
    message_to_be_sent += f"Position Size: {position.get('positionAmt')}\n"

webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/1febc77b-6b9d-4ed4-a7cc-6659a41c0aba"
bot = FeishuBot(webhook_url)

bot.send_text(message_to_be_sent)
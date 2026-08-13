from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, urlencode
from urllib.request import urlopen
import json, os, re

ORIGIN = os.getenv('PHASE1_ALLOWED_ORIGIN', 'https://samin110597-create.github.io')
SYMBOL = re.compile(r'^[A-Z0-9.\-]{1,10}$')

class handler(BaseHTTPRequestHandler):
    def reply(self, status, payload):
        body=json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Access-Control-Allow-Origin',ORIGIN)
        self.send_header('Cache-Control','no-store')
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        key=os.getenv('TWELVE_DATA_API_KEY')
        q=parse_qs(urlparse(self.path).query)
        symbol=(q.get('symbol',[''])[0] or '').upper().strip()
        if not key: return self.reply(503, {'ok':False,'error':'live feed not configured'})
        if not SYMBOL.fullmatch(symbol): return self.reply(400, {'ok':False,'error':'invalid ticker'})
        url='https://api.twelvedata.com/quote?'+urlencode({'symbol':symbol,'prepost':'true','apikey':key})
        try:
            data=json.loads(urlopen(url,timeout=8).read().decode())
            if data.get('status')=='error': return self.reply(502, {'ok':False,'error':data.get('message','provider error')})
            return self.reply(200, {'ok':True,'symbol':data.get('symbol',symbol),'price':float(data['close']),'quote_datetime':data.get('datetime'),'quote_timestamp':data.get('timestamp'),'exchange':data.get('exchange'),'is_extended_hours':bool(data.get('is_extended_hours',False)),'source':'Twelve Data'})
        except Exception:
            return self.reply(502, {'ok':False,'error':'quote unavailable'})

from flask import Flask, render_template, request, send_file, flash
from flask_wtf.csrf import CSRFProtect
from flask_wtf import FlaskForm
import io
import os
import re
import secrets
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import pandas as pd
import requests
import validators

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
csrf = CSRFProtect(app)

DEFAULT_ELEMENTS = ["p", "h1", "h2", "h3", "a"]


def parse_elements(raw: str) -> List[str]:
    if not raw:
        return DEFAULT_ELEMENTS
    elements: List[str] = []
    for part in raw.split(","):
        tag = part.strip().lower()
        if not tag:
            continue
        if not re.match(r"^[a-z0-9]+$", tag):
            continue
        elements.append(tag)
    return elements or DEFAULT_ELEMENTS

class ScraperForm(FlaskForm):
    pass

@app.route('/')
def home():
    form = ScraperForm()
    return render_template('index.html', form=form)

@app.route('/scrape', methods=['POST'])
def scrape():
    form = ScraperForm()
    if not form.validate_on_submit():
        flash('Invalid form submission', 'error')
        return render_template('index.html', form=form)

    try:
        url = request.form.get('url', '').strip()
        if not url or not validators.url(url):
            flash('Please provide a valid URL', 'error')
            return render_template('index.html', form=form)

        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            flash('URL must start with http:// or https://', 'error')
            return render_template('index.html', form=form)

        elements_raw = request.form.get('elements', '') or ''
        elements = parse_elements(elements_raw)

        data = scrape_website(url, elements)
        if not data:
            flash('No data found', 'error')
            return render_template('index.html', form=form)

        output = io.StringIO()
        pd.DataFrame(data).to_csv(output, index=False)
        
        domain = urlparse(url).netloc.replace('.', '_')
        filename = f"{domain}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return render_template('index.html', form=form)

def scrape_website(url: str, elements: Optional[List[str]] = None) -> List[Dict]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise ValueError(f"Failed to fetch URL: {exc}") from exc

    soup = BeautifulSoup(response.text, 'html.parser')
    data = []

    target_elements = elements or DEFAULT_ELEMENTS
    for element in soup.find_all(target_elements):
        text = element.get_text(strip=True)
        if text:
            record: Dict[str, str] = {
                'url': url,
                'text': text,
                'element': element.name,
                'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            if element.name == 'a':
                href = element.get('href')
                if href:
                    record['href'] = urljoin(url, href)
            data.append(record)
    
    return data

if __name__ == '__main__':
    app.run(debug=True)

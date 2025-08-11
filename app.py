from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import requests
from bs4 import BeautifulSoup
import pandas as pd
import io
from urllib.parse import urlparse, urljoin
import validators
from concurrent.futures import ThreadPoolExecutor
import re
from typing import Dict, List, Optional, Tuple
import os
import hashlib
import time
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

def is_valid_url(url):
    return validators.url(url)

def get_next_page_url(soup, base_url):
    # Common patterns for pagination links
    pagination_links = soup.find_all('a', href=True, text=re.compile(r'next|›|»|\d+', re.I))
    for link in pagination_links:
        href = link.get('href')
        if href and not href.startswith(('#', 'javascript')):
            return urljoin(base_url, href)
    return None

class RateLimiter:
    def __init__(self, requests_per_second=2):
        self.delay = 1.0 / requests_per_second
        self.last_request = 0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request = time.time()

class ContentExtractor:
    def __init__(self, selectors=None):
        self.selectors = selectors or {}
        self.default_elements = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'span', 'div', 'article', 'section']

    def get_target_elements(self):
        elements = self.selectors.get('elements', [])
        if elements and isinstance(elements, list) and len(elements) > 0:
            return [e for e in elements if e.strip()]
        return self.default_elements

    def extract_structured_data(self, element):
        text = element.get_text(strip=True)
        data = {
            'text': text,
            'element_type': element.name,
            'classes': ' '.join(element.get('class', [])),
            'id': element.get('id', ''),
            'href': element.get('href', '') if element.name == 'a' else '',
        }
        
        # Extract structured data patterns
        data['emails'] = ', '.join(re.findall(r'[\w\.-]+@[\w\.-]+\.[\w-]{2,}', text))
        data['phones'] = ', '.join(re.findall(r'\+?[\d\s\-\(\)]{10,}', text))
        data['prices'] = ', '.join(re.findall(r'[\$£€]\s*\d+(?:[,\.]\d{2})?', text))
        data['dates'] = ', '.join(re.findall(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', text))
        
        return data

class DataFilter:
    def __init__(self, keywords=None, exclude_words=None, min_length=0, max_length=None, patterns=None):
        self.keywords = [k.lower().strip() for k in (keywords or []) if k.strip()]
        self.exclude_words = [w.lower().strip() for w in (exclude_words or []) if w.strip()]
        self.min_length = max(0, min_length or 0)
        self.max_length = max_length if max_length and max_length > 0 else None
        self.patterns = patterns or {}

    def filter_content(self, text):
        if not text or not text.strip():
            return False
            
        text_lower = text.lower()
        text_length = len(text)
        
        # Length filters
        if self.min_length and text_length < self.min_length:
            return False
        if self.max_length and text_length > self.max_length:
            return False
            
        # Keyword filters
        if self.keywords:
            if not any(keyword in text_lower for keyword in self.keywords):
                return False
                
        # Exclude word filters
        if self.exclude_words:
            if any(word in text_lower for word in self.exclude_words):
                return False
                
        # Pattern filters
        if self.patterns:
            if self.patterns.get('email') and not re.search(r'[\w\.-]+@[\w\.-]+\.[\w-]{2,}', text):
                return False
            if self.patterns.get('phone') and not re.search(r'\+?[\d\s\-\(\)]{10,}', text):
                return False
            if self.patterns.get('price') and not re.search(r'[\$£€]\s*\d+(?:[,\.]\d{2})?', text):
                return False
            if self.patterns.get('date') and not re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', text):
                return False
            custom_pattern = self.patterns.get('custom', '').strip()
            if custom_pattern:
                try:
                    if not re.search(custom_pattern, text, re.I):
                        return False
                except re.error:
                    pass  # Invalid regex, ignore
                    
        return True

def scrape_page_enhanced(url: str, content_filter: DataFilter, extractor: ContentExtractor,
                        rate_limiter: RateLimiter = None) -> Tuple[List[Dict], Optional[str]]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Apply rate limiting
        if rate_limiter:
            rate_limiter.wait()
        
        print(f"Scraping URL: {url}")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        data = []
        
        # Get target elements to scrape
        target_elements = extractor.get_target_elements()
        print(f"Looking for elements: {target_elements}")
        
        # Find all specified elements
        all_elements = []
        for tag in target_elements:
            elements = soup.find_all(tag)
            if elements:
                all_elements.extend(elements)
                
        print(f"Found {len(all_elements)} total elements")
        
        # Process each element
        for element in all_elements:
            try:
                extracted_data = extractor.extract_structured_data(element)
                
                # Apply content filter
                if content_filter.filter_content(extracted_data['text']):
                    extracted_data['url'] = url
                    extracted_data['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    data.append(extracted_data)
            except Exception as e:
                print(f"Error processing element: {str(e)}")
                continue

        print(f"Extracted {len(data)} items after filtering")
        return data, get_next_page_url(soup, url)
        
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return [], None

def scrape_website(start_url: str, filters: Dict, max_pages: int = 5,
                  requests_per_second: int = 2) -> Tuple[Optional[io.StringIO], Optional[str]]:
    try:
        if not validators.url(start_url):
            raise ValueError("Invalid URL format")

        print(f"Starting scrape with filters: {filters}")
        
        # Initialize components
        content_filter = DataFilter(
            keywords=filters.get('keywords', []),
            exclude_words=filters.get('exclude_words', []),
            min_length=filters.get('min_length', 0),
            max_length=filters.get('max_length', None),
            patterns=filters.get('patterns', {})
        )
        
        extractor = ContentExtractor(filters.get('selectors', {}))
        rate_limiter = RateLimiter(requests_per_second)

        all_data = []
        visited_urls = set()
        current_url = start_url
        depth = 0

        while current_url and depth < max_pages and current_url not in visited_urls:
            print(f"\nProcessing page {depth + 1} of {max_pages}: {current_url}")
            visited_urls.add(current_url)
            
            page_data, next_url = scrape_page_enhanced(
                current_url, content_filter, extractor, rate_limiter
            )
            
            if page_data:
                all_data.extend(page_data)
                print(f"Total items collected so far: {len(all_data)}")
            else:
                print("No data found on this page")
                
            current_url = next_url
            depth += 1

        if not all_data:
            raise ValueError("No matching content found. Try adjusting your filters or removing some restrictions.")

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Ensure consistent column order
        base_columns = ['url', 'scraped_at', 'text', 'element_type', 'classes', 'id', 'href']
        extra_columns = ['emails', 'phones', 'prices', 'dates']
        
        ordered_columns = []
        for col in base_columns + extra_columns:
            if col in df.columns:
                ordered_columns.append(col)
        
        df = df[ordered_columns]

        # Create buffer
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        
        return csv_buffer, None
        
    except Exception as e:
        print(f"Scraping error: {str(e)}")
        return None, str(e)

def download_html(url: str, max_pages: int = 1) -> Tuple[Optional[io.BytesIO], Optional[str]]:
    try:
        if not validators.url(url):
            raise ValueError("Invalid URL format")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        all_html = []
        visited_urls = set()
        current_url = url
        depth = 0

        while current_url and depth < max_pages and current_url not in visited_urls:
            try:
                print(f"Downloading HTML from: {current_url}")
                response = requests.get(current_url, headers=headers, timeout=15)
                response.raise_for_status()
                
                page_html = f"\n<!-- Page URL: {current_url} -->\n{response.text}\n"
                all_html.append(page_html)
                
                if max_pages > 1:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    next_url = get_next_page_url(soup, current_url)
                    visited_urls.add(current_url)
                    current_url = next_url
                else:
                    break
                    
                depth += 1
                
            except Exception as e:
                print(f"Error downloading {current_url}: {str(e)}")
                break

        if not all_html:
            raise ValueError("Failed to download HTML content")

        combined_html = "\n\n".join(all_html)
        html_buffer = io.BytesIO(combined_html.encode('utf-8'))
        return html_buffer, None
        
    except Exception as e:
        return None, str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    url = request.form.get('url', '').strip()
    max_pages = int(request.form.get('max_pages', 5))
    requests_per_second = int(request.form.get('requests_per_second', 2))
    
    # Get element selectors
    elements_str = request.form.get('elements', '').strip()
    elements = [e.strip() for e in elements_str.split(',') if e.strip()] if elements_str else []
    
    # Build selectors dict
    selectors = {
        'elements': elements,
        'patterns': {
            'css': request.form.get('css_selector', '').strip(),
            'xpath': request.form.get('xpath_selector', '').strip()
        }
    }
    
    # Build filters dict
    keywords_str = request.form.get('keywords', '').strip()
    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()] if keywords_str else []
    
    exclude_str = request.form.get('exclude_words', '').strip()
    exclude_words = [w.strip() for w in exclude_str.split(',') if w.strip()] if exclude_str else []
    
    filters = {
        'keywords': keywords,
        'exclude_words': exclude_words,
        'min_length': int(request.form.get('min_length', 0) or 0),
        'max_length': int(request.form.get('max_length', 0) or 0) or None,
        'patterns': {
            'email': request.form.get('pattern_email') == 'on',
            'phone': request.form.get('pattern_phone') == 'on',
            'price': request.form.get('pattern_price') == 'on',
            'date': request.form.get('pattern_date') == 'on',
            'custom': request.form.get('pattern_custom', '').strip()
        },
        'selectors': selectors
    }
    
    if not url:
        flash('Please provide a URL', 'error')
        return render_template('index.html')
    
    try:
        domain = urlparse(url).netloc.replace('.', '_')
        filename = f"{domain}_scraped_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        csv_buffer, error = scrape_website(
            url, filters, max_pages, requests_per_second
        )
        
        if error:
            flash(error, 'error')
            return render_template('index.html')
        
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        flash(f"An error occurred: {str(e)}", 'error')
        return render_template('index.html')

@app.route('/download_html', methods=['POST'])
def download_html_route():
    url = request.form.get('url', '').strip()
    max_pages = int(request.form.get('max_pages', 1))
    
    if not url:
        flash('Please provide a URL', 'error')
        return render_template('index.html')
    
    try:
        domain = urlparse(url).netloc.replace('.', '_')
        filename = f"{domain}_raw_html.html" if domain else "raw_html.html"
        
        html_buffer, error = download_html(url, max_pages)
        
        if error:
            flash(error, 'error')
            return render_template('index.html')
        
        return send_file(
            html_buffer,
            mimetype='text/html',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        flash(f"An error occurred: {str(e)}", 'error')
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
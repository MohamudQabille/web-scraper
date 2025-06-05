from flask import Flask, render_template, request, send_file, flash
import requests
from bs4 import BeautifulSoup
import pandas as pd
import io
from urllib.parse import urlparse, urljoin
import validators
from concurrent.futures import ThreadPoolExecutor
import re
from typing import Dict, List, Optional, Tuple

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

def scrape_page(url, selectors=None, max_depth=3):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        data = []
        
        # Default elements to scrape if none specified
        default_elements = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'span', 'div', 'article', 'section']
        
        # Use provided elements or defaults
        target_elements = selectors.get('elements', []) if selectors and selectors.get('elements') else default_elements
        
        # Print debug information
        print(f"Scraping URL: {url}")
        print(f"Looking for elements: {target_elements}")
        
        # Find all specified elements
        elements = soup.find_all(target_elements)
        print(f"Found {len(elements)} matching elements")
        
        for element in elements:
            # Get text content
            text = element.get_text(strip=True)
            
            # Skip empty elements
            if not text:
                continue
                
            # Check if we have specific patterns to match
            if selectors and selectors.get('patterns'):
                css_selector = selectors['patterns'].get('css')
                xpath_selector = selectors['patterns'].get('xpath')
                
                # If we have patterns but none match, skip this element
                if css_selector and not element.select(css_selector):
                    continue
                if xpath_selector and not re.search(xpath_selector, text, re.I):
                    continue
            
            # Add the data
            data.append({
                'url': url,
                'content': text,
                'element_type': element.name,
                'class': ' '.join(element.get('class', [])),
                'id': element.get('id', '')
            })
        
        print(f"Extracted {len(data)} items with content")
        return data, get_next_page_url(soup, url) if max_depth > 1 else None
        
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return [], None

def scrape_website(start_url, selectors=None, max_pages=5):
    try:
        if not is_valid_url(start_url):
            raise ValueError("Invalid URL format")

        all_data = []
        visited_urls = set()
        current_url = start_url
        depth = 0

        while current_url and depth < max_pages and current_url not in visited_urls:
            print(f"\nProcessing page {depth + 1} of {max_pages}")
            visited_urls.add(current_url)
            page_data, next_url = scrape_page(current_url, selectors, max_pages - depth)
            
            if page_data:
                all_data.extend(page_data)
                print(f"Total items collected so far: {len(all_data)}")
            else:
                print("No data found on this page")
                
            current_url = next_url
            depth += 1

        if not all_data:
            raise ValueError("No content found on any of the pages. Try adjusting the element selectors or patterns.")

        # Convert to DataFrame and clean up
        df = pd.DataFrame(all_data)
        
        # Create buffer to store CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        
        return csv_buffer, None
    except Exception as e:
        return None, str(e)

def download_html(url: str, max_pages: int = 1) -> Tuple[Optional[io.BytesIO], Optional[str]]:
    try:
        if not validators.url(url):
            raise ValueError("Invalid URL format")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # Store HTML content from all pages
        all_html = []
        visited_urls = set()
        current_url = url
        depth = 0

        while current_url and depth < max_pages and current_url not in visited_urls:
            try:
                print(f"Downloading HTML from: {current_url}")
                response = requests.get(current_url, headers=headers, timeout=15)
                response.raise_for_status()
                
                # Add page URL as comment in HTML
                page_html = f"\n<!-- Page URL: {current_url} -->\n{response.text}\n"
                all_html.append(page_html)
                
                # Get next page if needed
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

        # Combine all HTML content
        combined_html = "\n\n".join(all_html)
        
        # Create buffer with HTML content
        html_buffer = io.BytesIO(combined_html.encode('utf-8'))
        return html_buffer, None
        
    except Exception as e:
        return None, str(e)

@app.route('/')
def home():
    return render_template('index.html')

# Add these imports at the top
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin, unquote
import os
import hashlib
import time
from datetime import datetime
from PIL import Image
from io import BytesIO

# Add these new classes after the existing ones
class ImageDownloader:
    def __init__(self, save_dir='downloaded_images'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def download_image(self, url, headers):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Verify it's an image
            img = Image.open(BytesIO(response.content))
            
            # Generate filename from URL
            filename = hashlib.md5(url.encode()).hexdigest()[:10]
            filename = f"{filename}.{img.format.lower()}"
            
            # Save image
            filepath = os.path.join(self.save_dir, filename)
            img.save(filepath)
            
            return filepath
        except Exception as e:
            print(f"Error downloading image {url}: {str(e)}")
            return None

class ProxyRotator:
    def __init__(self, proxies=None):
        self.proxies = proxies or []
        self.current_index = 0

    def add_proxy(self, proxy):
        self.proxies.append(proxy)

    def get_next_proxy(self):
        if not self.proxies:
            return None
        proxy = self.proxies[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.proxies)
        return proxy

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
        return self.selectors.get('elements', []) or self.default_elements

    def should_extract_element(self, element, text):
        if not text:
            return False

        if not self.selectors:
            return True

        if 'patterns' in self.selectors:
            css_selector = self.selectors['patterns'].get('css')
            xpath_selector = self.selectors['patterns'].get('xpath')

            if css_selector and not element.select(css_selector):
                return False
            if xpath_selector and not re.search(xpath_selector, text, re.I):
                return False

        return True

    def extract_data(self, element, url):
        return {
            'url': url,
            'content': element.get_text(strip=True),
            'element_type': element.name,
            'class': ' '.join(element.get('class', [])),
            'id': element.get('id', '')
        }

class DataFilter:
    def __init__(self, keywords=None, exclude_words=None, min_length=0, max_length=None, patterns=None):
        self.keywords = set(keywords) if keywords else set()
        self.exclude_words = set(exclude_words) if exclude_words else set()
        self.min_length = min_length
        self.max_length = max_length
        self.patterns = patterns or {}

    def matches_length(self, text):
        text_length = len(text)
        if self.min_length and text_length < self.min_length:
            return False
        if self.max_length and text_length > self.max_length:
            return False
        return True

    def matches_keywords(self, text):
        if not self.keywords:
            return True
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.keywords)

    def matches_exclude_words(self, text):
        if not self.exclude_words:
            return True
        text_lower = text.lower()
        return not any(word.lower() in text_lower for word in self.exclude_words)

    def matches_patterns(self, text):
        if not self.patterns:
            return True
        for pattern_type, pattern in self.patterns.items():
            if pattern_type == 'email' and pattern:
                if not re.search(r'[\w\.-]+@[\w\.-]+\.[\w-]{2,}', text):
                    return False
            elif pattern_type == 'phone' and pattern:
                if not re.search(r'\+?\d[\d\s-]{8,}\d', text):
                    return False
            elif pattern_type == 'price' and pattern:
                if not re.search(r'\$?\d+(?:\.\d{2})?', text):
                    return False
            elif pattern_type == 'date' and pattern:
                if not re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', text):
                    return False
            elif pattern_type == 'custom' and pattern:
                try:
                    if not re.search(pattern, text):
                        return False
                except re.error:
                    pass
        return True

    def filter_content(self, text):
        return (
            self.matches_length(text) and
            self.matches_keywords(text) and
            self.matches_exclude_words(text) and
            self.matches_patterns(text)
        )


def scrape_page(url: str, content_filter: DataFilter, extractor: ContentExtractor,
                download_images: bool = False, proxy: str = None,
                rate_limiter: RateLimiter = None) -> Tuple[List[Dict], Optional[str]]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Apply rate limiting
        if rate_limiter:
            rate_limiter.wait()
        
        # Setup proxy if provided
        proxies = {"http": proxy, "https": proxy} if proxy else None
        
        response = requests.get(url, headers=headers, proxies=proxies, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        data = []
        
        # Initialize image downloader if needed
        image_downloader = ImageDownloader() if download_images else None

        # Find all content elements
        target_elements = soup.find_all(['article', 'div', 'section', 'p', 'h1', 'h2', 'h3', 'a', 'img'],
                                      class_=re.compile(r'content|article|post|text|body', re.I))

        for element in target_elements:
            # Extract and filter content
            extracted_data = extractor.extract_structured_data(element)
            
            # Handle images
            if download_images and element.name == 'img':
                img_url = element.get('src')
                if img_url:
                    img_url = urljoin(url, img_url)
                    img_path = image_downloader.download_image(img_url, headers)
                    if img_path:
                        extracted_data['image_path'] = img_path

            if content_filter.matches(extracted_data['text']):
                extracted_data['url'] = url
                extracted_data['timestamp'] = datetime.now().isoformat()
                data.append(extracted_data)

        return data, get_next_page_url(soup, url)
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return [], None

# Update the scrape_website function
def scrape_website(start_url: str, filters: Dict, max_pages: int = 5,
                  download_images: bool = False, use_proxies: bool = False,
                  requests_per_second: int = 2) -> Tuple[Optional[io.StringIO], Optional[str]]:
    try:
        if not validators.url(start_url):
            raise ValueError("Invalid URL format")

        # Initialize components
        content_filter = DataFilter(
            keywords=filters.get('keywords', []),
            exclude_words=filters.get('exclude_words', []),
            min_length=filters.get('min_length', 0),
            max_length=filters.get('max_length', None)
        )
        extractor = ContentExtractor(filters.get('selectors', {}))
        rate_limiter = RateLimiter(requests_per_second)
        
        # Setup proxy rotation if enabled
        proxy_rotator = None
        if use_proxies:
            proxy_rotator = ProxyRotator([
                'http://proxy1:8080',
                'http://proxy2:8080'
                # Add more proxies as needed
            ])

        all_data = []
        visited_urls = set()
        current_url = start_url
        depth = 0

        while current_url and depth < max_pages and current_url not in visited_urls:
            visited_urls.add(current_url)
            proxy = proxy_rotator.get_next_proxy() if proxy_rotator else None
            
            page_data, next_url = scrape_page(
                current_url, content_filter, extractor,
                download_images=download_images,
                proxy=proxy,
                rate_limiter=rate_limiter
            )
            
            if page_data:
                all_data.extend(page_data)
            current_url = next_url
            depth += 1

        if not all_data:
            raise ValueError("No matching content found. Try adjusting your filters.")

        # Convert to DataFrame with enhanced columns
        df = pd.DataFrame(all_data)
        df['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Organize columns
        columns = ['url', 'scraped_at', 'text', 'element_type', 'classes', 'id', 'href',
                  'emails', 'phones', 'prices', 'dates', 'image_path', 'timestamp']
        df = df.reindex(columns=[col for col in columns if col in df.columns])

        # Create buffer
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        
        return csv_buffer, None
    except Exception as e:
        return None, str(e)

# Update the route to handle new features
@app.route('/scrape', methods=['POST'])
def scrape():
    url = request.form.get('url', '').strip()
    max_pages = int(request.form.get('max_pages', 5))
    download_images = request.form.get('download_images') == 'true'
    use_proxies = request.form.get('use_proxies') == 'true'
    requests_per_second = int(request.form.get('requests_per_second', 2))
    
    filters = {
        'keywords': [k.strip() for k in request.form.get('keywords', '').split(',') if k.strip()],
        'exclude_words': [w.strip() for w in request.form.get('exclude_words', '').split(',') if w.strip()],
        'min_length': int(request.form.get('min_length', 0)),
        'max_length': int(request.form.get('max_length', 0)) or None
    }
    
    if not url:
        flash('Please provide a URL', 'error')
        return render_template('index.html')
    
    try:
        domain = urlparse(url).netloc
        filename = f"{domain}_scraped_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        csv_buffer, error = scrape_website(
            url, filters, max_pages,
            download_images=download_images,
            use_proxies=use_proxies,
            requests_per_second=requests_per_second
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
        domain = urlparse(url).netloc
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
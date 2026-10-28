import os
import socket
import sys
import unittest
from concurrent.futures import Future

from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import HTTPException
from pydantic import ValidationError
from xgboost import XGBClassifier
import main
import feature_extraction as fe
from url_features import URL_MODEL_FEATURE_ORDER, extract_url_model_features
from feature_extraction import (
    DNSResolutionError,
    FetchTimeoutError,
    PageFetch,
    UnsafeTarget,
    WebsiteFetchError,
    _normalize_url,
    extract_features,
    fetch_page,
)

HTML = '''<!doctype html><html><head><title>Example page</title>
<meta name="description" content="test page"><meta name="viewport" content="width=device-width"></head>
<body><a href="/home">home</a><a href="https://outside.example/">outside</a>
<a href="">empty</a><a href="#">fragment</a>
<img src="image.png"><link rel="stylesheet" href="site.css"><script src="site.js"></script>
<iframe src="frame.html"></iframe><form action="https://outside.example/login">
<input type="password"><input type="hidden"><button>Submit</button></form></body></html>'''
PAGE = PageFetch('https://example.com/', HTML, 1, 0, 'text/html', [], 200)


class FakeResponse:
    def __init__(self, status_code, headers=None, body=b'', encoding='utf-8'):
        self.status_code = status_code
        self.headers = headers or {'Content-Type': 'text/html; charset=utf-8'}
        self.body = body
        self.encoding = encoding
        self.is_redirect = status_code in (301, 302, 303, 307, 308)
        self.is_permanent_redirect = status_code in (301, 308)
        self.closed = False

    def iter_content(self, size):
        for offset in range(0, len(self.body), size):
            yield self.body[offset:offset + size]

    def close(self):
        self.closed = True


class SpyModel:
    classes_ = [0, 1]

    def __init__(self, prediction=0):
        self.calls = 0
        self.prediction = prediction

    def predict(self, frame):
        self.calls += 1
        return [self.prediction]

    def predict_proba(self, frame):
        return [[0.8, 0.2]] if self.prediction == 0 else [[0.2, 0.8]]


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = main.load_runtime()
        cls.order = cls.loaded['order']
        cls.categorical = cls.loaded['categorical_features']

    def complete_features(self):
        return {key: ('example-value' if key in self.categorical else 1) for key in self.order}

    def predict_with(self, page, features=None, model=None, url_model=None):
        spy = model or SpyModel()
        extracted = features if features is not None else extract_features('https://example.com/', page)
        runtime = {**self.loaded, 'model': spy, 'full_model_loaded': True, 'url_model': url_model, 'url_model_loaded': url_model is not None}
        with patch.object(main, 'fetch_page', return_value=page), \
             patch.object(main, 'extract_features', return_value=extracted), \
             patch.object(main, 'runtime', runtime):
            result = main.predict(main.PredictRequest(url='https://example.com/'))
        return result, spy

    def test_saved_artifacts_and_feature_types_match(self):
        self.assertTrue(self.loaded['loaded'])
        self.assertEqual(len(self.order), 54)
        self.assertEqual(self.loaded['dataset_info']['feature_order'], self.order)
        self.assertEqual(set(self.categorical), {'URL', 'Domain', 'TLD', 'Title'})
        self.assertEqual(len(self.loaded['numerical_features']), 50)
        self.assertEqual(self.loaded['dataset_info']['label_mapping'], {0: 'Phishing', 1: 'Legitimate'})
        self.assertEqual(set(self.categorical) | set(self.loaded['numerical_features']), set(self.order))
        self.assertTrue(self.loaded['url_model_loaded'])
        self.assertIsInstance(self.loaded['url_model'].named_steps['classifier'], XGBClassifier)
        self.assertEqual(self.loaded['url_model_info']['model_version'], '2.0.0')
        self.assertEqual(tuple(self.loaded['url_model_info']['feature_order']), URL_MODEL_FEATURE_ORDER)

    def test_health_reports_ready_and_unavailable_http_statuses(self):
        with patch.object(main, 'runtime', self.loaded):
            ready = main.health()
        ready_payload = main.HealthResponse(**ready)
        self.assertEqual(ready_payload.status, 'ok')
        self.assertTrue(ready_payload.analysis_ready)

        with patch.object(main, 'runtime', {'loaded': False}):
            unavailable = main.health()
        self.assertIsInstance(unavailable, main.JSONResponse)
        self.assertEqual(unavailable.status_code, 503)
        self.assertIn(b'"status":"unavailable"', unavailable.body)
        self.assertNotIn(str(main.BASE_DIR).encode(), unavailable.body)
        with patch.object(main, 'runtime', {**self.loaded, 'url_model_loaded': False}):
            full_only = main.health()
        self.assertEqual(full_only.status_code, 503)
        self.assertIn(b'"model_loaded":true', full_only.body)
        self.assertNotIn(str(main.BASE_DIR).encode(), full_only.body)
        health_route = next(route for route in main.app.routes if getattr(route, 'path', None) == '/health')
        self.assertIn(503, health_route.responses)

    def test_rate_limit_returns_429_and_ignores_forwarded_ip_headers(self):
        limiter = main.InMemoryRateLimiter(limit=1, window_seconds=60)
        scope = {
            'type': 'http', 'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
            'path': '/predict', 'raw_path': b'/predict', 'query_string': b'',
            'headers': [(b'x-forwarded-for', b'203.0.113.99')],
            'client': ('198.51.100.23', 4567), 'server': ('testserver', 80),
        }
        request = main.Request(scope)
        with patch.object(main, 'rate_limiter', limiter):
            main.enforce_predict_rate_limit(request, main.Response())
            with self.assertRaises(HTTPException) as error:
                main.enforce_predict_rate_limit(request, main.Response())
        self.assertEqual(error.exception.status_code, 429)
        self.assertEqual(error.exception.headers['Retry-After'], '60')
        self.assertIn('rate limit exceeded', error.exception.detail.lower())
        predict_route = next(route for route in main.app.routes if getattr(route, 'path', None) == '/predict')
        dependency_calls = [dependency.call for dependency in predict_route.dependant.dependencies]
        self.assertIn(main.enforce_predict_rate_limit, dependency_calls)
        self.assertIn(429, predict_route.responses)

    def test_invalid_rate_limit_configuration_uses_safe_defaults(self):
        with patch.dict(os.environ, {'PHISHGUARD_RATE_LIMIT_REQUESTS': 'not-a-number'}):
            self.assertEqual(main._positive_env_int('PHISHGUARD_RATE_LIMIT_REQUESTS', 30, 10_000), 30)
        with patch.dict(os.environ, {'PHISHGUARD_RATE_LIMIT_WINDOW_SECONDS': '0'}):
            self.assertEqual(main._positive_env_int('PHISHGUARD_RATE_LIMIT_WINDOW_SECONDS', 60, 86_400), 60)
        with patch.dict(os.environ, {'PHISHGUARD_RATE_LIMIT_REQUESTS': '12'}):
            self.assertEqual(main._positive_env_int('PHISHGUARD_RATE_LIMIT_REQUESTS', 30, 10_000), 12)

    def test_model_loading_failure_is_reported(self):
        missing_dir = Path(main.BASE_DIR) / 'models-missing-test-fixture'
        with patch.object(main, 'MODEL_DIR', missing_dir):
            with self.assertRaisesRegex(RuntimeError, 'Required model artifact is missing'):
                main.load_runtime()
        with patch.object(main, 'runtime', {'loaded': False}):
            with self.assertRaises(HTTPException) as error:
                main.predict(main.PredictRequest(url='https://example.com/'))
        self.assertEqual(error.exception.status_code, 503)

    def test_url_model_load_error_preserves_original_54_feature_model(self):
        original_load = main.joblib.load
        def fail_url_artifact(path, *args, **kwargs):
            if Path(path).name == 'phishguard_url_model.joblib':
                raise ValueError('corrupt URL model fixture')
            return original_load(path, *args, **kwargs)
        with patch.object(main.joblib, 'load', side_effect=fail_url_artifact), self.assertLogs(main.logger, level='ERROR'):
            loaded = main.load_runtime()
        self.assertTrue(loaded['full_model_loaded'])
        self.assertFalse(loaded['url_model_loaded'])
        self.assertTrue(loaded['loaded'])

    def test_http_200_accessible_website_fetches_and_extracts_static_html(self):
        response = FakeResponse(200, body=HTML.encode())
        with patch.object(fe, '_resolve_public', return_value='93.184.216.34'), patch.object(fe, '_request_once', return_value=response):
            page = fetch_page('https://example.com/')
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.html, HTML)
        self.assertTrue(response.closed)
        features, missing, _ = extract_features('https://example.com/', page)
        self.assertEqual(len(features), 54)
        self.assertEqual(features['Title'], 'Example page')
        self.assertEqual(features['HasTitle'], 1)
        self.assertEqual(features['NoOfiFrame'], 1)
        self.assertEqual(features['HasPasswordField'], 1)
        self.assertEqual(features['HasExternalFormSubmit'], 1)
        self.assertIn('URLSimilarityIndex', missing)

    def test_403_uses_url_only_without_fetching_or_claiming_page_analysis(self):
        response = FakeResponse(403)
        with patch.object(fe, '_resolve_public', return_value='93.184.216.34'), patch.object(fe, '_request_once', return_value=response):
            page = fetch_page('https://example.com/')
        self.assertEqual(page.status_code, 403)
        self.assertEqual(page.html, '')
        url_model = SpyModel()
        result, model = self.predict_with(page, model=SpyModel(), url_model=url_model)
        self.assertEqual(result['analysis_status'], 'url_only')
        self.assertEqual(result['analysis_mode'], 'url_only')
        self.assertIsNotNone(result['prediction'])
        self.assertIsNotNone(result['confidence'])
        self.assertTrue(result['missing_features'])
        self.assertTrue(any('HTTP 403 denied page access' in warning and 'only the submitted URL' in warning for warning in result['warnings']))
        self.assertTrue(any('HTTP 403 (access denied)' in warning and 'separate URL-only model uses only the submitted URL' in warning for warning in result['warnings']))
        self.assertIn('no webpage values were imputed', result['imputation'])
        self.assertEqual(model.calls, 0)
        self.assertEqual(url_model.calls, 1)

    def test_403_actual_saved_url_model_produces_url_only_output(self):
        page = PageFetch('https://example.com/', '', 0, 0, 'text/html', ['HTTP 403 access denied'], 403)
        with patch.object(main, 'fetch_page', return_value=page), patch.object(main, 'runtime', self.loaded):
            result = main.predict(main.PredictRequest(url='https://example.com/'))
        main.PredictionResponse(**result)
        self.assertEqual(result['analysis_status'], 'url_only')
        self.assertEqual(result['model_name'], 'phishguard_url_only')
        self.assertEqual(result['model_version'], '2.0.0')
        self.assertIn(result['prediction'], ('phishing', 'legitimate'))
        self.assertIsInstance(result['confidence'], float)
        self.assertEqual(result['features_extracted'], 19)
        self.assertGreater(len(result['missing_features']), 0)
        self.assertTrue(any('HTTP 403 denied page access' in warning for warning in result['warnings']))

    def test_401_uses_url_only_without_bypassing_page_access(self):
        response = FakeResponse(401)
        with patch.object(fe, '_resolve_public', return_value='93.184.216.34'), patch.object(fe, '_request_once', return_value=response):
            page = fetch_page('https://example.com/')
        url_model = SpyModel()
        result, model = self.predict_with(page, model=SpyModel(), url_model=url_model)
        self.assertEqual(result['analysis_status'], 'url_only')
        self.assertEqual(result['analysis_mode'], 'url_only')
        self.assertIsNotNone(result['prediction'])
        self.assertIsNotNone(result['confidence'])
        self.assertGreater(result['features_extracted'], 0)
        self.assertTrue(any('HTTP 401 (access denied)' in warning for warning in result['warnings']))
        self.assertEqual(model.calls, 0)
        self.assertEqual(url_model.calls, 1)

    def test_url_only_pipeline_runs_for_accessible_partial_analysis(self):
        url_model = SpyModel(prediction=1)
        result, full_model = self.predict_with(PAGE, model=SpyModel(), url_model=url_model)
        self.assertEqual(result['analysis_status'], 'url_only')
        self.assertEqual(result['analysis_mode'], 'url_only')
        main.PredictionResponse(**result)
        self.assertEqual(result['model_name'], 'phishguard_url_only')
        self.assertEqual(result['label'], 1)
        self.assertEqual(result['label_name'], 'Legitimate')
        self.assertEqual(result['prediction'], 'legitimate')
        self.assertEqual(result['confidence'], 0.8)
        self.assertEqual(set(result['url_model_features']), set(main.URL_MODEL_FEATURE_ORDER))
        self.assertTrue(any('not equivalent to full webpage analysis' in warning for warning in result['warnings']))
        self.assertEqual(full_model.calls, 0)
        self.assertEqual(url_model.calls, 1)

    def test_actual_url_only_pipeline_runs_and_returns_finite_probability(self):
        runtime = {**self.loaded, 'model': SpyModel(), 'full_model_loaded': True}
        with patch.object(main, 'fetch_page', return_value=PAGE), patch.object(main, 'runtime', runtime):
            result = main.predict(main.PredictRequest(url='https://example.com/'))
        self.assertEqual(result['analysis_mode'], 'url_only')
        self.assertIn(result['label'], (0, 1))
        self.assertEqual(result['label_name'], self.loaded['dataset_info']['label_mapping'][result['label']])
        self.assertTrue(0.0 <= result['confidence'] <= 1.0)
        self.assertEqual(runtime['model'].calls, 0)
        with patch.object(main, 'fetch_page', return_value=PAGE), patch.object(main, 'runtime', runtime):
            repeated = main.predict(main.PredictRequest(url='HTTPS://EXAMPLE.COM'))
        self.assertEqual(result['prediction'], repeated['prediction'])
        self.assertAlmostEqual(result['confidence'], repeated['confidence'], places=12)

    def test_incomplete_features_use_url_model_and_do_not_call_full_model(self):
        url_model = SpyModel(prediction=1)
        result, model = self.predict_with(PAGE, model=SpyModel(), url_model=url_model)
        self.assertEqual(result['analysis_status'], 'url_only')
        self.assertLess(result['features_extracted'], result['features_total'])
        self.assertIsNotNone(result['prediction'])
        self.assertIsNotNone(result['confidence'])
        self.assertIn('URLSimilarityIndex', result['missing_features'])
        self.assertIsNone(result['extracted_features']['URLSimilarityIndex'])
        self.assertEqual(model.calls, 0)
        self.assertEqual(url_model.calls, 1)

    def test_only_complete_54_valid_features_run_saved_model(self):
        for label, name in ((0, 'Phishing'), (1, 'Legitimate')):
            with self.subTest(label=label):
                result, model = self.predict_with(PAGE, (self.complete_features(), [], []), SpyModel(label))
                self.assertEqual(result['analysis_status'], 'complete')
                self.assertEqual(result['label'], label)
                self.assertEqual(result['label_name'], name)
                self.assertEqual(result['prediction'], 'phishing' if label == 0 else 'legitimate')
                self.assertEqual(result['confidence'], 0.8)
                self.assertEqual(model.calls, 1)

    def test_full_model_takes_precedence_when_all_54_features_are_valid(self):
        url_model = SpyModel(prediction=1)
        result, full_model = self.predict_with(PAGE, (self.complete_features(), [], []), SpyModel(prediction=0), url_model=url_model)
        self.assertEqual(result['analysis_mode'], 'full_webpage')
        self.assertEqual(result['label'], 0)
        self.assertEqual(full_model.calls, 1)
        self.assertEqual(url_model.calls, 0)

    def test_complete_features_run_the_actual_saved_pipeline(self):
        features = self.complete_features()
        with patch.object(main, 'fetch_page', return_value=PAGE), \
             patch.object(main, 'extract_features', return_value=(features, [], [])), \
             patch.object(main, 'runtime', self.loaded):
            result = main.predict(main.PredictRequest(url='https://example.com/'))
        self.assertEqual(result['analysis_status'], 'complete')
        self.assertEqual(result['analysis_mode'], 'full_webpage')
        main.PredictionResponse(**result)
        self.assertEqual(result['model_name'], 'phiusiil_54_feature')
        self.assertIn(result['label'], (0, 1))
        self.assertEqual(result['label_name'], self.loaded['dataset_info']['label_mapping'][result['label']])
        self.assertIn(result['prediction'], ('phishing', 'legitimate'))
        self.assertIsInstance(result['confidence'], float)
        self.assertEqual(result['model_version'], 'original-54-feature')

    def test_incompatible_feature_type_blocks_inference(self):
        values = self.complete_features()
        values['URLLength'] = 'not-a-number'
        values['URL'] = 'https://example.com/'
        result, model = self.predict_with(PAGE, (values, [], []), SpyModel(), url_model=SpyModel(prediction=1))
        url_model = SpyModel(prediction=1)
        result, model = self.predict_with(PAGE, (values, [], []), SpyModel(), url_model=url_model)
        self.assertEqual(result['analysis_status'], 'url_only')
        self.assertIn('URLLength', result['missing_features'])
        self.assertIsNone(result['extracted_features']['URLLength'])
        self.assertIsNotNone(result['confidence'])
        self.assertEqual(model.calls, 0)
        self.assertEqual(url_model.calls, 1)

    def test_feature_order_tld_and_url_normalization_edge_cases(self):
        url = 'HTTPS://WWW.Shop.Example.co.uk/path%2Fpart?q=1&n=2#frag'
        normalized = _normalize_url(url)
        self.assertEqual(normalized, 'https://www.shop.example.co.uk/path%2Fpart?q=1&n=2#frag')
        features, _, _ = extract_features(url, PAGE)
        self.assertEqual(features['URL'], normalized)
        self.assertEqual(features['URLLength'], len(normalized))
        self.assertEqual(features['Domain'], 'www.shop.example.co.uk')
        self.assertEqual(features['TLD'], 'uk')
        self.assertEqual(features['TLDLength'], 2)
        self.assertEqual(features['NoOfSubDomain'], 3)
        self.assertEqual(features['NoOfQMarkInURL'], 1)
        self.assertEqual(features['NoOfAmpersandInURL'], 1)
        self.assertEqual(features['NoOfEqualsInURL'], 2)
        live_url_features = extract_url_model_features(normalized)
        self.assertEqual(tuple(live_url_features), URL_MODEL_FEATURE_ORDER)
        self.assertEqual(live_url_features['tld'], 'uk')
        self.assertEqual(live_url_features['host_length'], len('www.shop.example.co.uk'))
        self.assertEqual(live_url_features['query_parameter_count'], 2)
        ordinary, _, _ = extract_features('https://example.com', PAGE)
        self.assertEqual(ordinary['URL'], 'https://example.com/')
        self.assertEqual(ordinary['TLD'], 'com')
        self.assertEqual(ordinary['TLDLength'], 3)

    def test_ip_url_representation_and_ipv6_policy(self):
        for value in ('http://127.0.0.1/', 'http://192.168.1.2/', 'http://169.254.169.254/', 'http://[::1]/', 'http://[fc00::1]/'):
            with self.subTest(value=value), self.assertRaises(UnsafeTarget):
                _normalize_url(value)
        self.assertEqual(_normalize_url('https://[2606:4700:4700::1111]'), 'https://[2606:4700:4700::1111]/')
        ip_features, _, _ = extract_features('https://93.184.216.34/', PAGE)
        self.assertEqual(ip_features['IsDomainIP'], 1)
        self.assertIsNone(ip_features['TLD'])
        self.assertIsNone(ip_features['TLDLength'])

    def test_invalid_schemes_hosts_and_malformed_urls_are_rejected(self):
        for value in ('ftp://example.com/', 'http://localhost/', 'http://service.internal/', 'http://metadata.google.internal/', 'http://[broken/', 'http://bad host/', 'http://user:pass@example.com/', 'http://example.com:8080/'):
            with self.subTest(value=value), self.assertRaises(UnsafeTarget):
                _normalize_url(value)
        with self.assertRaises(ValidationError):
            main.PredictRequest(url='')
        with self.assertRaises(ValidationError):
            main.PredictRequest(url='https://example.com', extra='forbidden')
        url_model = SpyModel()
        with patch.object(main, 'fetch_page', side_effect=UnsafeTarget('private host')), \
             patch.object(main, 'runtime', {**self.loaded, 'url_model': url_model, 'url_model_loaded': True}):
            with self.assertRaises(HTTPException) as error:
                main.predict(main.PredictRequest(url='https://example.com/'))
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(url_model.calls, 0)

    def test_dns_resolution_errors_and_private_dns_answers_are_distinguished(self):
        with patch.object(fe.socket, 'getaddrinfo', side_effect=socket.gaierror('no record')):
            with self.assertRaises(DNSResolutionError):
                fe._resolve_public('example.com', 443)
        future = Future()
        future.set_result([(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 443))])
        with patch.object(fe._DNS_EXECUTOR, 'submit', return_value=future):
            with self.assertRaises(UnsafeTarget):
                fe._resolve_public('example.com', 443)
        class PendingFuture:
            def result(self, timeout):
                raise fe.FutureTimeoutError()
            def cancel(self):
                return True
        with patch.object(fe._DNS_EXECUTOR, 'submit', return_value=PendingFuture()):
            with self.assertRaises(FetchTimeoutError):
                fe._resolve_public('example.com', 443)

    def test_localhost_private_and_cloud_metadata_are_blocked(self):
        targets = (
            'http://localhost/', 'http://127.0.0.1/', 'http://10.0.0.1/',
            'http://192.168.0.1/', 'http://169.254.169.254/latest/meta-data/',
            'http://metadata.google.internal/', 'http://instance-data.ec2.internal/',
        )
        for target in targets:
            with self.subTest(target=target), self.assertRaises(UnsafeTarget):
                _normalize_url(target)

    def test_redirect_to_private_or_metadata_host_is_blocked(self):
        for destination in ('http://127.0.0.1/latest/meta-data', 'http://metadata.google.internal/'):
            redirect = FakeResponse(302, {'Location': destination})
            with self.subTest(destination=destination), patch.object(fe, '_resolve_public', return_value='93.184.216.34'), patch.object(fe, '_request_once', return_value=redirect):
                with self.assertRaises(UnsafeTarget):
                    fetch_page('https://example.com/')

    def test_redirect_limit_and_https_downgrade_are_rejected(self):
        redirect = FakeResponse(302, {'Location': '/next'})
        with patch.object(fe, '_resolve_public', return_value='93.184.216.34'), patch.object(fe, '_request_once', return_value=redirect):
            with self.assertRaisesRegex(UnsafeTarget, 'excessive redirect'):
                fetch_page('https://example.com/')
        downgrade = FakeResponse(302, {'Location': 'http://example.com/next'})
        with patch.object(fe, '_resolve_public', return_value='93.184.216.34'), patch.object(fe, '_request_once', return_value=downgrade):
            with self.assertRaisesRegex(UnsafeTarget, 'downgrades are blocked'):
                fetch_page('https://example.com/')

    def test_timeout_types_are_distinct(self):
        with patch.object(fe, '_request_once', side_effect=TimeoutError('timed out')):
            with self.assertRaises(FetchTimeoutError):
                fetch_page('https://example.com/')
        with patch.object(fe, '_request_once', side_effect=fe.DNSResolutionError('no DNS')):
            with self.assertRaises(DNSResolutionError):
                fetch_page('https://example.com/')
        url_model = SpyModel(prediction=0)
        with patch.object(main, 'fetch_page', side_effect=FetchTimeoutError('timed out')), \
             patch.object(main, 'runtime', {**self.loaded, 'model': SpyModel(), 'full_model_loaded': True, 'url_model': url_model, 'url_model_loaded': True}):
            result = main.predict(main.PredictRequest(url='https://example.com/'))
        self.assertEqual(result['analysis_mode'], 'url_only')
        self.assertEqual(result['label'], 0)
        self.assertEqual(result['prediction'], 'phishing')
        self.assertTrue(0.0 <= result['confidence'] <= 1.0)
        self.assertEqual(url_model.calls, 1)

    def test_oversized_response_is_bounded_and_becomes_partial(self):
        response = FakeResponse(200, {'Content-Type': 'text/html', 'Content-Length': str(fe.MAX_HTML_BYTES + 1)}, b'x' * 20)
        with patch.object(fe, '_resolve_public', return_value='93.184.216.34'), patch.object(fe, '_request_once', return_value=response):
            page = fetch_page('https://example.com/')
        self.assertEqual(page.html, '')
        self.assertTrue(any('exceeded the download limit' in warning for warning in page.warnings))
        self.assertTrue(response.closed)
        streamed = FakeResponse(200, {'Content-Type': 'text/html'}, b'x' * (fe.MAX_HTML_BYTES + 1))
        with patch.object(fe, '_resolve_public', return_value='93.184.216.34'), patch.object(fe, '_request_once', return_value=streamed):
            page2 = fetch_page('https://example.com/')
        self.assertEqual(page2.html, '')
        self.assertTrue(streamed.closed)

    def test_non_html_and_http_errors_are_not_misclassified_as_phishing(self):
        response = FakeResponse(200, {'Content-Type': 'application/pdf'}, b'%PDF')
        with patch.object(fe, '_resolve_public', return_value='93.184.216.34'), patch.object(fe, '_request_once', return_value=response):
            page = fetch_page('https://example.com/')
        no_url_runtime = {**self.loaded, 'model': SpyModel(), 'full_model_loaded': True, 'url_model': None, 'url_model_loaded': False}
        with patch.object(main, 'fetch_page', return_value=page), patch.object(main, 'extract_features', return_value=extract_features('https://example.com/', page)), patch.object(main, 'runtime', no_url_runtime):
            with self.assertRaises(HTTPException) as error:
                main.predict(main.PredictRequest(url='https://example.com/'))
        self.assertEqual(error.exception.status_code, 503)
        for status in (404, 429, 500, 503):
            response = FakeResponse(status)
            with self.subTest(status=status), patch.object(fe, '_resolve_public', return_value='93.184.216.34'), patch.object(fe, '_request_once', return_value=response):
                page_error = fetch_page('https://example.com/')
            url_model = SpyModel(prediction=1)
            result, full_model = self.predict_with(page_error, model=SpyModel(), url_model=url_model)
            self.assertEqual(result['analysis_status'], 'url_only')
            self.assertEqual(result['analysis_mode'], 'url_only')
            self.assertEqual(result['label'], 1)
            self.assertEqual(full_model.calls, 0)
            self.assertEqual(url_model.calls, 1)

    def test_empty_successful_html_response_is_partial(self):
        response = FakeResponse(200, body=b'')
        with patch.object(fe, '_resolve_public', return_value='93.184.216.34'), patch.object(fe, '_request_once', return_value=response):
            page = fetch_page('https://example.com/')
        url_model = SpyModel(prediction=0)
        result, model = self.predict_with(page, model=SpyModel(), url_model=url_model)
        self.assertEqual(result['analysis_status'], 'url_only')
        self.assertIsNotNone(result['prediction'])
        self.assertIsNotNone(result['confidence'])
        self.assertIn('Title', result['missing_features'])
        self.assertEqual(model.calls, 0)
        self.assertEqual(url_model.calls, 1)

    def test_malformed_html_is_parsed_without_executing_scripts(self):
        malformed = '<html><title>Broken</title><meta name=description><form action=javascript:alert(1)><input type=password>'
        page = PageFetch('https://example.com/', malformed, 0, 0, 'text/html', [], 200)
        features, _, _ = extract_features('https://example.com/', page)
        self.assertEqual(features['HasTitle'], 1)
        self.assertEqual(features['HasPasswordField'], 1)
        self.assertEqual(features['NoOfJS'], 0)

    def test_health_and_prediction_openapi_schemas_exist(self):
        spec = main.app.openapi()
        self.assertIn('/health', spec['paths'])
        self.assertIn('/predict', spec['paths'])
        self.assertIn('PredictionResponse', spec['components']['schemas'])
        with patch.object(main, 'runtime', self.loaded):
            health = main.HealthResponse(**main.health())
        self.assertTrue(health.analysis_ready)
        self.assertTrue(health.model_loaded)
        self.assertTrue(health.url_model_loaded)


if __name__ == '__main__':
    unittest.main()

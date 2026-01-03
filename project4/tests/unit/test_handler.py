import pytest
import sys
import os
import json
from unittest.mock import MagicMock, patch

# Add lambda directory to path to avoid syntax error importing 'lambda'
sys.path.append(os.path.join(os.path.dirname(__file__), '../../lambda'))

# Mock env vars
os.environ['LAMBDA_TASK_ROOT'] = 'dummy_root'

# We need to mock boto3 before importing handler because it initializes a client at module level
with patch('boto3.client') as mock_boto:
    import handler

def test_handler_success():
    # Setup S3 mock
    mock_s3 = MagicMock()
    handler.s3_client = mock_s3
    
    # Mock download_file to just create a dummy file
    def side_effect_download(bucket, key, path):
        # Create directory if needed for /tmp
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write("dummy image content")
    mock_s3.download_file.side_effect = side_effect_download

    # Mock predict function
    with patch('handler.predict') as mock_predict:
        mock_predict.return_value = 281 # Tabby cat ID in ImageNet
        
        event = {
            'Records': [
                {
                    's3': {
                        'bucket': {'name': 'test-bucket'},
                        'object': {'key': 'cute_cat.jpg'}
                    }
                }
            ]
        }
        
        response = handler.lambda_handler(event, None)
        
        # Verify response
        assert response['statusCode'] == 200
        
        # Verify download called
        # The path arg is random uuid, so we check bucket and key
        args, _ = mock_s3.download_file.call_args
        assert args[0] == 'test-bucket'
        assert args[1] == 'cute_cat.jpg'
        
        # Verify predict called
        mock_predict.assert_called_once()
        
        # Verify put_object called (result upload)
        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs['Bucket'] == 'test-bucket'
        assert call_kwargs['Key'] == 'cute_cat.jpg_result.json'
        
        body = json.loads(call_kwargs['Body'])
        assert body['class_id'] == 281
        assert body['source_image'] == 'cute_cat.jpg'

def test_handler_skip_result_file():
    # Setup S3 mock
    mock_s3 = MagicMock()
    handler.s3_client = mock_s3
    
    event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'test-bucket'},
                    'object': {'key': 'existing_result.json_result.json'}
                }
            }
        ]
    }
    
    # Should not throw
    handler.lambda_handler(event, None)
    
    # Should skip download and predict
    mock_s3.download_file.assert_not_called()
    mock_s3.put_object.assert_not_called()

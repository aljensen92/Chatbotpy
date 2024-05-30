from flask import Flask, request, jsonify
import requests
import json
import time
import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging
from threading import Lock

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

load_dotenv()  # Load environment variables from .env file

API_URL = 'https://api.openai.com/v1'
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
API_KEY = os.getenv("API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_ADMIN_MEMBER_ID = os.getenv("SLACK_ADMIN_MEMBER_ID")

slack_client = WebClient(token=SLACK_BOT_TOKEN)

# A simple file-based cache to store processed event IDs and a mutex lock
PROCESSED_EVENTS_FILE = 'processed_events.json'
event_lock = Lock()

def load_processed_events():
    try:
        if os.path.exists(PROCESSED_EVENTS_FILE):
            with open(PROCESSED_EVENTS_FILE, 'r') as f:
                return set(json.load(f))
    except Exception as e:
        logging.error(f"Error loading processed events: {e}")
    return set()

def save_processed_events(events):
    try:
        with open(PROCESSED_EVENTS_FILE, 'w') as f:
            json.dump(list(events), f)
    except Exception as e:
        logging.error(f"Error saving processed events: {e}")

processed_events = load_processed_events()

def create_thread(input_text):
    logging.info(f"Creating thread for input text: {input_text}")
    thread_data = {
        "assistant_id": ASSISTANT_ID,
        "thread": {
            "messages": [
                {"role": "user", "content": input_text}
            ]
        }
    }
    response = requests.post(f'{API_URL}/threads/runs',
                             headers={
                                 'Content-Type': 'application/json',
                                 'Authorization': f'Bearer {API_KEY}',
                                 'OpenAI-Beta': 'assistants=v2'
                             },
                             data=json.dumps(thread_data))
    if response.status_code != 200:
        logging.error(f"Failed to create thread: {response.status_code}, {response.text}")
        raise Exception(f"Failed to create thread: {response.status_code}, {response.text}")
    logging.info(f"Thread created successfully: {response.json()}")
    return response.json()

def wait_for_run_completion(thread_id):
    while True:
        run_response = requests.get(f'{API_URL}/threads/{thread_id}/runs',
                                    headers={
                                        'Content-Type': 'application/json',
                                        'Authorization': f'Bearer {API_KEY}',
                                        'OpenAI-Beta': 'assistants=v2'
                                    })
        run_data = run_response.json()
        if run_data['data'][0]['status'] in ['completed', 'failed', 'incomplete']:
            logging.info(f"Run completed with status: {run_data['data'][0]['status']}")
            return run_data['data'][0]['status']
        logging.info("Waiting for run to complete...")
        time.sleep(5)

def get_thread_messages(thread_id):
    logging.info(f"Getting messages for thread ID: {thread_id}")
    response = requests.get(f'{API_URL}/threads/{thread_id}/messages',
                            headers={
                                'Content-Type': 'application/json',
                                'Authorization': f'Bearer {API_KEY}',
                                'OpenAI-Beta': 'assistants=v2'
                            })
    if response.status_code != 200:
        logging.error(f"Failed to get thread messages: {response.status_code}, {response.text}")
        raise Exception(f"Failed to get thread messages: {response.status_code}, {response.text}")
    logging.info(f"Thread messages retrieved successfully: {response.json()}")
    return response.json()

def format_links(text):
    # Replace markdown links with Slack-compatible links
    formatted_text = text
    while True:
        start_idx = formatted_text.find('[')
        end_idx = formatted_text.find(')', start_idx)
        if start_idx == -1 or end_idx == -1:
            break
        link_text = formatted_text[start_idx+1:formatted_text.find(']', start_idx)]
        url = formatted_text[formatted_text.find('(', start_idx)+1:end_idx]
        slack_link = f"<{url}|{link_text}>"
        formatted_text = formatted_text[:start_idx] + slack_link + formatted_text[end_idx+1:]
    return formatted_text

def get_last_message(channel, thread_ts):
    try:
        logging.info(f"Fetching last message from channel {channel}, thread {thread_ts}")
        response = slack_client.conversations_replies(
            channel=channel,
            ts=thread_ts
        )
        if response['messages']:
            last_message = response['messages'][-1]['text']
            logging.info(f"Last message: {last_message}")
            return last_message
    except SlackApiError as e:
        logging.error(f"Error fetching last message: {e.response['error']}")
    return None

def send_gpt_response(channel, thread_ts, text):
    formatted_text = format_links(text)
    logging.info(f"Preparing to send response to channel {channel}, thread {thread_ts}: {formatted_text}")
    last_message = get_last_message(channel, thread_ts)

    if last_message != formatted_text:
        try:
            response = slack_client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=formatted_text
            )
            logging.info(f"Message sent to Slack: {response['message']['text']}")
        except SlackApiError as e:
            error_message = f"Error sending message to Slack: {e.response['error']}"
            logging.error(error_message)
            slack_client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=f"<@{SLACK_ADMIN_MEMBER_ID}> {error_message}"
            )
    else:
        logging.info(f"Duplicate message detected, not sending: {formatted_text}")

@app.route('/slack', methods=['POST'])
def slack():
    data = request.get_json()
    logging.info(f"Received request: {json.dumps(data, indent=2)}")

    if data.get('type') == 'url_verification':
        challenge = data.get('challenge')
        return jsonify({"challenge": challenge})

    event = data.get('event', {})
    event_id = data.get('event_id')
    user_message = event.get('text', '')
    channel_id = event.get('channel')
    thread_ts = event.get('ts')
    retry_num = request.headers.get('X-Slack-Retry-Num')

    logging.info(f"Processing event {event_id} from channel {channel_id}, thread {thread_ts}")

    # Quick acknowledgement to Slack
    if retry_num:
        logging.info(f"Retry request received: {retry_num}")
        return jsonify({"status": "ok"})

    # Check if the event has already been processed
    if event_id in processed_events:
        logging.info(f"Event {event_id} already processed")
        return jsonify({"status": "ok", "message": "Event already processed"})

    with event_lock:
        # Mark the event as processed
        processed_events.add(event_id)
        save_processed_events(processed_events)

        try:
            thread_run = create_thread(user_message)
            thread_id = thread_run.get('thread_id')

            # Wait for the run to complete
            status = wait_for_run_completion(thread_id)

            if status == 'completed':
                messages = get_thread_messages(thread_id)

                # Log the full response for debugging purposes
                logging.info(f"API response: {json.dumps(messages, indent=2)}")

                # Ensure we have the expected structure in the response
                if 'data' in messages and messages['data']:
                    last_message_content = messages['data'][0].get('content')
                    if last_message_content and isinstance(last_message_content, list) and 'text' in last_message_content[0]:
                        last_message = last_message_content[0]['text']['value']
                    else:
                        raise Exception("Unexpected API response structure")
                else:
                    raise Exception("Unexpected API response structure")

                send_gpt_response(channel_id, thread_ts, last_message)
                return jsonify({"status": "ok"})
            else:
                error_message = f"Run failed with status: {status}"
                send_gpt_response(channel_id, thread_ts, error_message)
                return jsonify({"status": "error", "message": error_message})
        except Exception as e:
            error_message = f"Error: {str(e)}"
            send_gpt_response(channel_id, thread_ts, error_message)
            return jsonify({"status": "error", "message": error_message})

if __name__ == '__main__':
    app.run(debug=True, port=8080)

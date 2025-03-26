# PERFECT, duplicate issue fixed, auto mapping extensions fixed, auto call n times tracker fixed
# FIX prompt later, sentiment cannot be NA it has to be a number. 
import subprocess
import speech_recognition as sr
import os
from pydub import AudioSegment
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
import nltk
nltk.download('punkt_tab')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException
from datetime import datetime, timedelta
import time
import re
import json
from google import genai
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Download the vader_lexicon resource if it's not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    print("Downloading vader_lexicon...")
    nltk.download('vader_lexicon')

# Download the stopwords resource if it's not already downloaded
try:
    nltk.data.find('stopwords')
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')

# Download the punkt resource if it's not already downloaded
try:
    nltk.data.find('punkt')
    import nltk
    nltk.download('punkt_tab')
except LookupError:
    print("Downloading punkt...")
    import nltk
    nltk.download('punkt_tab')
    nltk.download('punkt')

def convert_mp3_to_wav(mp3_file_path):
    print(f"Converting MP3 file to WAV: {mp3_file_path}")
    # Convert MP3 to WAV
    wav_file_path = mp3_file_path.replace(".mp3", ".wav")
    ffmpeg_path = r"C:\ffmpeg\ffmpeg.exe"
    command = f'"{ffmpeg_path}" -i "{mp3_file_path}" "{wav_file_path}"'
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)
    print(f"Converted MP3 file to WAV: {wav_file_path}")
    return wav_file_path

def transcribe_audio(filename):
    print(f"Transcribing audio file: {filename}")
    """Transcribe audio file using Google Speech Recognition."""
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        print(f"Recording audio from file: {filename}")
        audio_data = r.record(source)
        try:
            print(f"Recognizing speech from audio data...")
            text = r.recognize_google(audio_data)
            print(f"Transcribed text: {text}")
            return text
        except sr.UnknownValueError as e:
            print(f"Error (Google Speech Recognition): {str(e)}")
            return ""

def get_large_audio_transcription_fixed_interval(path, seconds=60):
    print(f"Splitting audio into chunks and transcribing each chunk: {path}")
    """Split audio into chunks and transcribe each chunk."""
    sound = AudioSegment.from_file(path)
    chunk_length_ms = int(1000 * seconds)
    chunks = [sound[i:i + chunk_length_ms] for i in range(0, len(sound), chunk_length_ms)]
    folder_name = "audio-fixed-chunks"
    if not os.path.isdir(folder_name):
        print(f"Creating folder: {folder_name}")
        os.mkdir(folder_name)
    whole_text = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        print(f"Processing chunk {i} of {len(chunks)}")
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        print(f"Exporting chunk to file: {chunk_filename}")
        audio_chunk.export(chunk_filename, format="wav")
        text = transcribe_audio(chunk_filename)
        if text:
            text = f"{text.capitalize()}. "
            print(f"Transcribed text from chunk {i}: {text}")
            whole_text += text
    print(f"Finished transcribing all chunks. Whole text: {whole_text}")
    return whole_text

def send_to_gemini(text):
    print(f"Sending text to Gemini: {text}")
    api_key = ""
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Role: Eyes Now Optometrist Office Assistant

    Context: Eyes Now is an optometrist office with 4 locations in Dallas, DFW, Hurst, and Southlake. Patients call us for various reasons such as booking appointments, asking general questions, insurance questions, prescription questions, and more.

    Task: As an expert and experienced call center Quality Assurance Manager overeeing agents, Study the transcription of the call, understand it, and generate a smart summary that includes who called(try to get the name of the patient if possible), why they called, how they were assisted, and the solution provided. Also, include any follow-up tasks to be done by the agent and recommendations to improve for the agent.

    Output Format:

    Summary: 
    - (write summary point 1) 
    - (write summary point 2) 
    - (write summary point 3)  
    - (write summary point 4) 
    - (write summary point 5) 
    
    To-Do:
    - (write task point 1) 
    - (write task point 2) 

    Sentiment Score: (1-5) (sentiment should be an integer, cannot be NA, you must assign a score. If Tag is Missed Call then sentiment is 0)
    Feedback: How could we have better handled this call or conversation with the patient? (answer as a single liner not bulleted points and if Tag is Missed Call Don't give any feedback just say "None as this call was missed")

    TAG: (here you will assign this call a tag that represents what occurred, for example if you see that the transcript mostly contains the IVR message and the patient never really talked with anyone then that would mean its's a Missed Call, you have to choose only 1 from one of these tags:
    Appointment Confirmed, Appointment Rescheduled, Appointment Cancelled, Complaint, Missed Call, Billing Query, Rx Query, Contact Lenses Query, Location Query, Business Calls ( use business calls tag when someone from a different company calls in to request a referral or check in if their patient came in or request fax information), Other)

    (NOTE:also make sure that you do not bold anything, so that when i copy the entire message and paste it, it won't be pasted with those 2 stars)

    Text: {text}
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite", contents=prompt
    )
    print(f"Result from Gemini: {response.text}")
    return response.text

def load_processed_files():
    print("Loading processed files...")
    """Load processed files from JSON file."""
    processed_files_path = r"C:\mango_VMs_Download\processed_files.json"
    try:
        with open(processed_files_path, 'r') as f:
            data = json.load(f)
            # Convert lists to sets
            processed_files = {}
            for date, files in data.items():
                processed_files[date] = set(files)
            print(f"Loaded processed files: {processed_files}")
            return processed_files
    except FileNotFoundError:
        print("No processed files found.")
        return {}

def save_processed_files(processed_files):
    print("Saving processed files...")
    """Save processed files to JSON file."""
    processed_files_path = r"C:\mango_VMs_Download\processed_files.json"
    # Convert sets to lists for JSON serialization
    data = {}
    for date, files in processed_files.items():
        data[date] = list(files)
    with open(processed_files_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved processed files: {data}")

def load_call_attempts():
    print("Loading call attempts...")
    call_attempts_path = r"C:\mango_VMs_Download\CallAttempts.json"
    try:
        with open(call_attempts_path, 'r') as f:
            data = json.load(f)
            print(f"Loaded call attempts: {data}")
            return data
    except FileNotFoundError:
        print("No call attempts found. Creating new file...")
        with open(call_attempts_path, 'w') as f:
            json.dump({}, f)
        return {}

def save_call_attempts(call_attempts):
    print("Saving call attempts...")
    call_attempts_path = r"C:\mango_VMs_Download\CallAttempts.json"
    with open(call_attempts_path, 'w') as f:
        json.dump(call_attempts, f, indent=4)
    print(f"Saved call attempts: {call_attempts}")

def update_call_attempts(call_attempts, source_number, direction, disposition):
    print(f"Updating call attempts for source number: {source_number}")
    today = datetime.now().strftime("%Y-%m-%d")
    if today not in call_attempts:
        call_attempts[today] = {}
    if source_number not in call_attempts[today]:
        if direction == "Inbound" and disposition in ["Hangup", "Rejected", "Voicemail"]:
            call_attempts[today][source_number] = 1
        else:
            call_attempts[today][source_number] = 0
    else:
        if direction == "Inbound" and disposition in ["Hangup", "Rejected", "Voicemail"]:
            call_attempts[today][source_number] += 1
        else:
            call_attempts[today][source_number] = 0
    return call_attempts

def get_call_attempt(call_attempts, source_number):
    print(f"Getting call attempt for source number: {source_number}")
    today = datetime.now().strftime("%Y-%m-%d")
    if today in call_attempts and source_number in call_attempts[today]:
        return call_attempts[today][source_number]
    else:
        return 0

def get_answered_by(extension):
    print(f"Getting answered by for extension: {extension}")
    """Get the name of the person who answered the call based on the extension."""
    extension_map = {
        "777": "ACCOUNTS STACY",
        "901": "ANDRES",
        "902": "ASHWINI",
        "601": "BILLING SHARED VOICEMAIL",
        "606": "CALL CENTER SHARED VOICEMAIL",
        "900": "CALL QUEUE",
        "301": "DALLAS LINE 1",
        "302": "DALLAS LINE 2",
        "303": "DALLAS LINE 3",
        "451": "DALLAS RING GROUP",
        "603": "DALLAS SHARED VOICEMAIL",
        "1006": "DALLAS SWITCHBOARD",
        "1008": "DALLAS VM CALLBACKS",
        "101": "DFW LINE 1",
        "102": "DFW LINE 2",
        "200": "DFW RING GROUP",
        "605": "DFW SHARED VOICEMAIL",
        "1000": "DFW SWITCHBOARD",
        "500": "DFW VIRTUAL FAX",
        "1010": "DFW VM CALLBACKS",
        "800": "EXECUTIVE ASSISTANT",
        "201": "HURST LINE 1",
        "202": "HURST LINE 2",
        "452": "HURST RING GROUP",
        "604": "HURST SHARED VOICEMAIL",
        "1007": "HURST SWITCHBOARD",
        "1011": "HURST VM CALLBACKS",
        "905": "JESSICA HOJILLA",
        "906": "JOYCE MYRELL",
        "904": "KATRINA",
        "903": "KRISHNAN",
        "450": "PED RING GROUP",
        "602": "PED SHARED VOICEMAIL",
        "1005": "PED SWITCHBOARD",
        "501": "PRIMARY EYE DOCTOR",
        "999": "PRIMARY EYE QUEUE",
        "401": "SOUTHLAKE LINE 1",
        "402": "SOUTHLAKE LINE 2",
        "403": "SOUTHLAKE LINE 3",
        "400": "SOUTHLAKE RING GROUP",
        "600": "SOUTHLAKE SHARED VOICEMAIL",
        "1001": "SOUTHLAKE SWITCHBOARD",
        "1009": "SOUTHLAKE VM CALLBACKS",
        "541": "VIRTUAL FAX"
    }
    print(f"Answered by: {extension_map.get(extension, 'None')}")
    return extension_map.get(extension, "None")

def login_to_mangovoice():
    print("Logging in to MangoVoice...")
    # Set up the webdriver
    driver = webdriver.Chrome() # Replace with your preferred browser
    driver.maximize_window() # Maximize the Chrome window

    try:
        # Navigate to the login page
        print("Navigating to login page...")
        driver.get("https://admin.mangovoice.com/user/login")

        # Enter username and password
        print("Entering username and password...")
        username_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "user_email"))
        )
        username_input.send_keys("")

        password_input = driver.find_element(By.ID, "user_password")
        password_input.send_keys("")

        # Click the login button
        print("Clicking login button...")
        login_button = driver.find_element(By.ID, "user_login")
        login_button.click()

        # Wait for the plus icon to be clickable
        print("Waiting for plus icon to be clickable...")
        plus_icon = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-target='logs-sm']"))
        )
        plus_icon.click()

        # Wait for the Legacy Call Logs link to be clickable
        print("Waiting for Legacy Call Logs link to be clickable...")
        legacy_call_logs_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.LINK_TEXT, "Legacy Call Logs"))
        )
        legacy_call_logs_link.click()

        # Wait for the start date box to be clickable
        print("Waiting for start date box to be clickable...")
        start_date_box = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "dateFrom"))
        )
        start_date_box.click()

        # Clear the start date box
        print("Clearing start date box...")
        start_date_box.send_keys("\b" * 10) # Send backspace keys to clear the box

        # Get today's date
        print("Getting today's date...")
        today = datetime.now()
        today_date = today.strftime("%m/%d/%Y")

        # Enter today's date in the start date box
        print("Entering today's date in start date box...")
        start_date_box.send_keys(today_date)

        # Wait for the end date box to be clickable
        print("Waiting for end date box to be clickable...")
        end_date_box = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "dateTo"))
        )
        end_date_box.click()

        # Clear the end date box
        print("Clearing end date box...")
        end_date_box.send_keys("\b" * 10) # Send backspace keys to clear the box

        # Enter today's date in the end date box
        print("Entering today's date in end date box...")
        end_date_box.send_keys(today_date)

        # Wait for the search button to be clickable
        print("Waiting for search button to be clickable...")
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".cdrLogsSearchButton"))
        )
        search_button.click()

        # Wait for 5 seconds
        print("Waiting for 5 seconds...")
        time.sleep(5)

        # Wait for the display dropdown to be clickable
        print("Waiting for display dropdown to be clickable...")
        display_dropdown = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.NAME, "listLogs_length"))
        )
        display_dropdown.click()

        # Select the option value 500
        print("Selecting option value 500...")
        options = display_dropdown.find_elements(By.TAG_NAME, "option")
        for option in options:
            if option.get_attribute("value") == "500":
                option.click()
                break

        # Click on the "LEGACY CALL LOGS" heading
        print("Clicking on LEGACY CALL LOGS heading...")
        legacy_call_logs_heading = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "h3.sub-title.inline.mango-default-heading.pull-left"))
        )
        legacy_call_logs_heading.click()

        # Wait for 5 seconds
        print("Waiting for 5 seconds...")
        time.sleep(5)

        # Extract the total number of entries available
        print("Extracting total number of entries available...")
        total_entries_div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "listLogs_info"))
        )
        total_entries_text = total_entries_div.text
        total_entries = int(total_entries_text.split("of ")[1].split(" entries")[0])
        print(f"Total number of entries available: {total_entries}")

        # Create a new folder based on the current date
        print("Creating new folder based on current date...")
        folder_path = f"C:\\mango_VMs_Download\\{today_date.replace('/', '_')}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Keep the Chrome window open for 10 seconds before extracting row data
        print("Keeping Chrome window open for 10 seconds...")
        time.sleep(10)

        # Load processed files
        print("Loading processed files...")
        processed_files = load_processed_files()

        # Initialize the processed_files for today if not exists
        if today_date not in processed_files:
            print("Initializing processed files for today...")
            processed_files[today_date] = set()

        # Load call attempts
        print("Loading call attempts...")
        call_attempts = load_call_attempts()

        # Extract the source number, destination number, destination extension, duration, and audio src link for each row
        print("Extracting row data...")
        rows = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tr[role='row']"))
        )
        i = 0
        file_audio_url_map = {}
        call_time_map = {}
        while i < total_entries:
            try:
                print(f"Processing row {i+1} of {total_entries}...")
                columns = rows[i].find_elements(By.TAG_NAME, "td")
                if len(columns) < 11:
                    print(f"Row {i+1} does not have enough columns. Skipping...")
                    i += 1
                    continue
                call_time = columns[0].text
                direction = columns[1].text
                source_number = columns[2].text
                destination_number = columns[5].text
                destination_extension = columns[6].text
                duration = columns[7].text
                disposition = columns[8].text
                time_to_answer = columns[10].text

                # Reformat the call time
                call_time = call_time.replace("th", "").replace("st", "").replace("nd", "").replace("rd", "")
                call_time = datetime.strptime(call_time, "%B %d %Y %I:%M %p")
                call_time = call_time.strftime("%m/%d/%Y %I:%M %p")

                # Check for duplicate rows
                row_key = (source_number, destination_number, duration, destination_extension)
                if row_key in processed_files.get(today_date, set()):
                    print(f"Skipping duplicate row {i+1}: {row_key}")
                    i += 1
                    continue

                # Extract the audio src link from the last column
                audio_src_link = None
                last_column = columns[-1]
                audio_tag = last_column.find_elements(By.TAG_NAME, "audio")
                if audio_tag:
                    source_tag = audio_tag[0].find_elements(By.TAG_NAME, "source")
                    if source_tag:
                        audio_src_link = source_tag[0].get_attribute("src")
                        if audio_src_link and not audio_src_link.startswith("https:"):
                            audio_src_link = None

                if audio_src_link:
                    # Download the audio file
                    print(f"Downloading audio file from {audio_src_link}...")
                    filename = f"SrcNo_{source_number}_Ext{destination_extension}.mp3"
                    if filename not in processed_files.get(today_date, set()):
                        file_path = os.path.join(folder_path, filename)
                        response = requests.get(audio_src_link, stream=True)
                        if response.status_code == 200:
                            with open(file_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=1024):
                                    if chunk:
                                        f.write(chunk)
                            print(f"File saved to {file_path}")
                            file_audio_url_map[filename] = audio_src_link
                            call_time_map[filename] = call_time
                        else:
                            print(f"Failed to download file from {audio_src_link}")
                    else:
                        print(f"Skipping already processed file: {filename}")
                else:
                    print("Audio Src Link not found or invalid")

                print(f"Row {i+1}:")
                print(f"Call Time = {call_time}")
                print(f"Direction = {direction}")
                print(f"Source Number = {source_number}")
                print(f"Destination Number = {destination_number}")
                print(f"Destination Extension = {destination_extension}")
                print(f"Duration = {duration}")
                print(f"Disposition = {disposition}")
                print(f"Time to Answer = {time_to_answer}")
                if audio_src_link:
                    print(f"Audio Src Link = {audio_src_link}")
                else:
                    print("Audio Src Link not found or invalid")

                # Remove the '+' character from the source number
                source_number = source_number.replace('+', '')

                # Update call attempts
                call_attempts = update_call_attempts(call_attempts, source_number, direction, disposition)
                save_call_attempts(call_attempts) # Save call attempts immediately

                # Log data to Google Sheets
                log_to_google_sheets(call_time, direction, source_number, destination_number, destination_extension, duration, disposition, time_to_answer, audio_src_link)

                i += 1
                if i % 8 == 0: # Scroll down every 8 rows
                    print("Scrolling down to next rows...")
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2) # Wait for the next rows to load
                    rows = WebDriverWait(driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tr[role='row']"))
                    )
            except Exception as e:
                print(f"An error occurred: {e}")
                i += 1

        # Close the Chrome window after all rows have been downloaded
        print("Closing Chrome window...")
        driver.quit()

        print("Converting MP3 files to WAV...")

        # Convert MP3 files to WAV and delete the original MP3 files
        for filename in os.listdir(folder_path):
            print(f"Processing file: {filename}")
            if filename.endswith(".mp3"):
                file_path = os.path.join(folder_path, filename)
                wav_file_path = convert_mp3_to_wav(file_path)
                os.remove(file_path)
                print(f"Converted {filename} to WAV and deleted the original MP3 file: {wav_file_path}")

                # Analyze the WAV file and send the analysis to Gemini
                transcribed_text = get_large_audio_transcription_fixed_interval(wav_file_path, seconds=15)
                print("Transcribed Text:\n", transcribed_text)

                if not transcribed_text:
                    print("No text was transcribed from the audio. Skipping this file.")
                    # Add file to processed files
                    if today_date not in processed_files:
                        processed_files[today_date] = set()
                    processed_files[today_date].add(filename)
                    save_processed_files({today_date: processed_files[today_date]}) # Save only the updated date
                    os.remove(wav_file_path)
                    print(f"Deleted WAV file: {wav_file_path}")
                    continue

                result = send_to_gemini(transcribed_text)
                print(f"Result from Gemini: {result}")

                # Extract info from Slack message
                answered_by_slack, sentiment_score, tag, call_attempt_slack = extract_info_from_slack_message(result)
                print(f"Extracted info from Slack message: Answered By = {answered_by_slack}, Sentiment Score = {sentiment_score}, TAG = {tag}, Call Attempt = {call_attempt_slack}")

                # Send Slack message
                slack_webhook_url = ""
                audio_url = file_audio_url_map[filename]
                extension = filename.split("_Ext")[1].split(".mp3")[0]
                answered_by = get_answered_by(extension)
                call_attempts = load_call_attempts()
                today = datetime.now().strftime("%Y-%m-%d")
                source_number = filename.split("SrcNo_")[1].split("_Ext")[0].replace("+", "")
                call_attempt = call_attempts.get(today, {}).get(source_number, 0)
                message = {
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Call Summary - {filename}*"
                            }
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Answered By:* {answered_by}"
                            }
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Call Details:* \n{result}"
                            }
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Call Link:* <{audio_url}|Download>"
                            }
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Call Attempt:* {call_attempt}"
                            }
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*----------------------------------------🛑----------------------------------------*"
                            }
                        }
                    ]
                }
                response = requests.post(slack_webhook_url, json=message)
                if response.status_code == 200:
                    print("Message sent to Slack successfully.")
                else:
                    print("Error sending message to Slack.")

                # Update Google Sheet
                spreadsheet_id = ''
                call_time = call_time_map[filename]
                print(f"Searching for call time: {call_time} in Google Sheet...")
                update_google_sheet(spreadsheet_id, call_time, filename, answered_by_slack, sentiment_score, tag, call_attempt_slack, file_audio_url_map[filename])

                # Add file to processed files
                if today_date not in processed_files:
                    processed_files[today_date] = set()
                processed_files[today_date].add(filename)
                save_processed_files({today_date: processed_files[today_date]}) # Save only the updated date

                # Delete the WAV file after processing
                os.remove(wav_file_path)
                print(f"Deleted WAV file: {wav_file_path}")
            else:
                print(f"Skipping non-MP3 file: {filename}")
        print("Finished processing all files.")

        # Save the entire processed_files dictionary
        save_processed_files(processed_files)

    except TimeoutException:
        print("Timed out waiting for page to load")
    finally:
        # Close the browser window if an exception occurs
        try:
            driver.quit()
        except NameError:
            pass

def extract_info_from_slack_message(message):
    print("Extracting info from Slack message...")
    answered_by = None
    sentiment_score = None
    tag = None
    call_attempt = None
    lines = message.split("\n")
    for line in lines:
        if "Answered By:" in line:
            answered_by = line.split(":")[1].strip()
        elif "Sentiment Score:" in line:
            sentiment_score = line.split(":")[1].strip()
        elif "TAG:" in line:
            tag = line.split(":")[1].strip()
        elif "Call Attempt:" in line:
            call_attempt = line.split(":")[1].strip()
    return answered_by, sentiment_score, tag, call_attempt

def update_google_sheet(spreadsheet_id, call_time, filename, answered_by, sentiment_score, tag, call_attempt, audio_url):
    print("Updating Google Sheet...")
    credentials = service_account.Credentials.from_service_account_info(
        {
        """ enter creds here """
        }
    )

    service = build('sheets', 'v4', credentials=credentials)

    spreadsheet_id = ''

    # Remove leading zero from hour
    call_time_no_zero = call_time.split(" ")[0] + " " + call_time.split(" ")[1].lstrip('0') + " " + call_time.split(" ")[2]

    # Get the data from the Google Sheet
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range='Sheet1!A:M'
    ).execute()
    values = result.get('values', [])

    # Find the row to update
    row_to_update = None
    src_number = filename.split("SrcNo_")[1].split("_Ext")[0].replace("+", "")
    for i, value in enumerate(values):
        if value and value[0] == call_time_no_zero and value[2] == src_number:
            row_to_update = i + 1
            break

    if row_to_update is not None:
        # Update the row
        range_name = f"Sheet1!I{row_to_update}:L{row_to_update}"
        body = {
            'values': [[answered_by, sentiment_score, tag, call_attempt]]
        }
        result = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id, range=range_name,
            valueInputOption='USER_ENTERED', body=body).execute()
        print(f"Updated Google Sheet: {result}")
    else:
        print(f"Row not found in Google Sheet.")

def log_to_google_sheets(call_time, direction, source_number, destination_number, destination_extension, duration, disposition, time_to_answer, audio_src_link):
    print("Logging data to Google Sheets...")
    # Set up credentials
    credentials = service_account.Credentials.from_service_account_info(
        {
            """ enter creds here """
        }
    )

    # Set up Google Sheets API client
    service = build('sheets', 'v4', credentials=credentials)

    # Set up spreadsheet ID and range
    spreadsheet_id = ''
    range_name = 'Sheet1!A1:M1'

    # Determine the answered_by value based on the extension
    extension_map = {
        "777": "ACCOUNTS STACY",
        "901": "ANDRES",
        "902": "ASHWINI",
        "601": "BILLING SHARED VOICEMAIL",
        "606": "CALL CENTER SHARED VOICEMAIL",
        "900": "CALL QUEUE",
        "301": "DALLAS LINE 1",
        "302": "DALLAS LINE 2",
        "303": "DALLAS LINE 3",
        "451": "DALLAS RING GROUP",
        "603": "DALLAS SHARED VOICEMAIL",
        "1006": "DALLAS SWITCHBOARD",
        "1008": "DALLAS VM CALLBACKS",
        "101": "DFW LINE 1",
        "102": "DFW LINE 2",
        "200": "DFW RING GROUP",
        "605": "DFW SHARED VOICEMAIL",
        "1000": "DFW SWITCHBOARD",
        "500": "DFW VIRTUAL FAX",
        "1010": "DFW VM CALLBACKS",
        "800": "EXECUTIVE ASSISTANT",
        "201": "HURST LINE 1",
        "202": "HURST LINE 2",
        "452": "HURST RING GROUP",
        "604": "HURST SHARED VOICEMAIL",
        "1007": "HURST SWITCHBOARD",
        "1011": "HURST VM CALLBACKS",
        "905": "JESSICA HOJILLA",
        "906": "JOYCE MYRELL",
        "904": "KATRINA",
        "903": "KRISHNAN",
        "450": "PED RING GROUP",
        "602": "PED SHARED VOICEMAIL",
        "1005": "PED SWITCHBOARD",
        "501": "PRIMARY EYE DOCTOR",
        "999": "PRIMARY EYE QUEUE",
        "401": "SOUTHLAKE LINE 1",
        "402": "SOUTHLAKE LINE 2",
        "403": "SOUTHLAKE LINE 3",
        "400": "SOUTHLAKE RING GROUP",
        "600": "SOUTHLAKE SHARED VOICEMAIL",
        "1001": "SOUTHLAKE SWITCHBOARD",
        "1009": "SOUTHLAKE VM CALLBACKS",
        "541": "VIRTUAL FAX"
    }

    answered_by = extension_map.get(destination_extension, "")

    # Get call attempt from CallAttempts.json
    call_attempt = 0
    try:
        # Load call attempts
        call_attempts_path = r"C:\mango_VMs_Download\CallAttempts.json"
        with open(call_attempts_path, 'r') as f:
            call_attempts = json.load(f)

        # Get today's date in the format used in CallAttempts.json
        today = datetime.now().strftime("%Y-%m-%d")

        # Check if today's date exists in call_attempts
        if today in call_attempts:
            # Clean up the source number to match the format in CallAttempts.json
            clean_source_number = source_number.strip()

            # Check if the source number exists in today's call attempts
            if clean_source_number in call_attempts[today]:
                call_attempt = call_attempts[today][clean_source_number]
                print(f"Found call attempt for {clean_source_number}: {call_attempt}")
            else:
                print(f"Source number {clean_source_number} not found in today's call attempts")
        else:
            print(f"Today's date {today} not found in call attempts")
    except Exception as e:
        print(f"Error getting call attempt: {e}")

    # Set up data to log with the answered_by value in column I (index 8) and call_attempt in column L (index 11)
    data = [
        [call_time, direction, source_number, destination_number, destination_extension, duration, disposition, time_to_answer, answered_by, "", "", str(call_attempt), audio_src_link]
    ]

    # Remove leading zero from hour
    call_time_no_zero = call_time.split(" ")[0] + " " + call_time.split(" ")[1].lstrip('0') + " " + call_time.split(" ")[2]

    # Check if the row already exists
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range='Sheet1!A:M'
    ).execute()
    values = result.get('values', [])

    row_exists = False
    for value in values:
        if value and value[0] == call_time_no_zero and value[2] == source_number:
            row_exists = True
            break

    if not row_exists:
        # Log data to Google Sheets
        body = {
            'values': data
        }
        result = service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id, range=range_name,
            valueInputOption='USER_ENTERED', body=body).execute()
        print(f"Logged data to Google Sheets: {result}")
    else:
        print(f"Row already exists in Google Sheet. Skipping...")

def job():
    print("Starting job...")
    login_to_mangovoice()
    print("Job finished.")

if __name__ == "__main__":
    now = datetime.now()
    next_run_time = now.replace(minute=33, second=0)
    if now > next_run_time:
        next_run_time += timedelta(hours=1)

    print(f"Next run time: {next_run_time}")

    while True:
        now = datetime.now()
        if now >= next_run_time:
            print("Running job...")
            job()
            next_run_time += timedelta(hours=1)
        time.sleep(1)
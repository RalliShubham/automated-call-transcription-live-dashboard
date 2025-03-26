# Automated-Call-Transcription-Live-Dashboard
This project manages call logs, their associated files, transcribes them and pushes them into a pipeline for sentiment analysis, summarization, caller identification, call tagging, agent analysis, and sending key details for QA on slack where all the data is logged in the backend and can be seen on a Looker Dashboard updating live.

The Live Dashboard can be seen at: https://lookerstudio.google.com/reporting/ae47276d-3998-46d1-98d2-8c8c06536927
## Overview
The Automated Call Log Analysis System is designed to automate the extraction, transcription, and analysis of call log data for customer support centers. The system uses Python and Selenium for web scraping, integrates speech-to-text APIs for call transcription, and applies machine learning techniques to predict call outcomes and analyze sentiment. The final results are visualized on a live Looker dashboard, providing real-time insights into customer interactions.

This project aims to improve the efficiency of customer support operations by automating tedious manual tasks and providing actionable insights into customer sentiment and support quality.

## Features
- Call Log Extraction & Management: The system scrapes call logs from a specified website using Python and Selenium. This eliminates the need for manual data retrieval and accelerates the process of collecting historical call data.

- Call Transcription: Recorded calls are transcribed into text using advanced speech-to-text APIs. This enables in-depth analysis of the customer support conversations.

- Sentiment Analysis & Call Tagging: The system performs sentiment analysis on the transcribed text, categorizing calls into positive, neutral, or negative sentiments. It also tags the calls with keywords (e.g., "refund," "product issue," "technical support") to help identify common issues and topics.

- Outcome Prediction: The system uses machine learning models to predict the likelihood of successful issue resolution and customer satisfaction based on historical data from previous calls.

- Real-time Analytics with Looker: The results from sentiment analysis, tagging, and predictions are displayed on a dynamic Looker dashboard. This dashboard provides real-time insights into customer service operations, helping managers make data-driven decisions.

- Sends a live message via incoming webhooks to slack too!

## Technologies Used
- Python: Core language for scraping, data processing, and machine learning model development.
- Selenium: Web scraping tool used to extract call logs and manage data.
- Speech-to-Text APIs: Used for transcribing recorded support calls into textual data.
- Looker: Business intelligence tool used to visualize data and display real-time insights on a dashboard.
- Machine Learning: Regression models for outcome prediction and sentiment analysis.

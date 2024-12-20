import os
import json
import datetime
import csv
import random
import nltk
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from plyer import notification
from dateutil.parser import parse

# SSL context for nltk download
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Initialize vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Store tasks in a CSV file
TASK_FILE = "tasks.csv"

# Check if the task file exists, if not, create it
if not os.path.exists(TASK_FILE):
    with open(TASK_FILE, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Task', 'Deadline', 'Priority'])  # Header
def get_task_priority(deadline):
    current_time = datetime.datetime.now()
    time_remaining = deadline - current_time

    if time_remaining.total_seconds() < 86400:  # Less than 24 hours
        return "High"
    elif time_remaining.total_seconds() < 259200:  # Between 1 and 3 days (24*3 hours = 72 hours)
        return "Medium"
    else:
        return "Low"
# Function to add a task to CSV file
def add_task(task, deadline_str, priority="Medium"):
    try:
        if "today" in deadline_str.lower():
            deadline = datetime.datetime.now()
        elif "tomorrow" in deadline_str.lower():
            deadline = datetime.datetime.now() + datetime.timedelta(days=1)
        else:
            deadline = parse(deadline_str)  # Parse the deadline using dateparser
        priority = get_task_priority(deadline)
        task_data = [task, deadline_str, priority]
        
        with open(TASK_FILE, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(task_data)
        return f"Task added with {priority} priority!"  # Inform the user that the task was added
    
    except Exception as e:
        return f"Error adding task: {str(e)}"

# Function to retrieve tasks sorted by deadline


def get_next_task():
    tasks = []
    with open(TASK_FILE, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        tasks = sorted(list(reader), key=lambda x: parse(x[1]))  # Sort by deadline using dateparser.parse
    
    if tasks:
        next_task = tasks[0]  # First task is the next one to do
        task_name = next_task[0]
        deadline_str = next_task[1]
        
        # Parse the deadline using dateparser instead of datetime.strptime
        deadline = parse(deadline_str)  # This can handle formats like "Friday at 2 PM"
        
        if deadline is None:
            return "Sorry, I couldn't understand the deadline format."
        
        time_remaining = deadline - datetime.datetime.now()

        # Automatically assign priority based on the time remaining
        priority=get_task_priority(deadline)
        # Send notification if the task is due soon (within 5 minutes)
        if time_remaining.total_seconds() < 300:  # 5 minutes
            send_local_notification(task_name, str(time_remaining))
        
        # Return task information with priority
        return f"Your next task is: {task_name} with deadline {next_task[1]} and priority {priority}"
    else:
        return "You have no upcoming tasks!"


# Function to send a local notification
def send_local_notification(task_name, time_remaining):
    notification.notify(
        title=f"Upcoming Task: {task_name}",
        message=f"You have {time_remaining} left to complete this task.",
        timeout=10  # Notification duration in seconds
    )
def get_initial_response():
    # Look for the default or greeting tag in the JSON file
    for intent in intents:
        if intent['tag'] == 'default':  # Change 'default' to any tag you use for initial response
            return random.choice(intent['responses'])  # Return the first response from the "responses" list
    return "Hello! How can I assist you today?" 

# Function to predict user input and return response
def chatbot(input_text):
    
   # Check if the input has the "next task" keyword indicating a request for next task
    if "next task" in input_text.lower() or "tasks" in input_text.lower():
        task_response = get_next_task()
        initial_response = get_initial_response()
        return initial_response + "\n\n" + task_response
    
    elif  " by " in input_text or " at " in input_text:
        if " at " in input_text:
            task_info = input_text.split(" at ")
        if " by " in input_text:
            task_info = input_text.split(" by ")
        if len(task_info) == 2:
            task = task_info[0].strip()
            deadline = task_info[1].strip()
            task_response = add_task(task, deadline)
            return task_response  # Inform the user that the task was added
    
    # If no task information is found, process it for classification (using the vectorizer)
    input_text_transformed = vectorizer.transform([input_text])
    tag = clf.predict(input_text_transformed)[0]  # Predict the intent based on the input

    for intent in intents:
        if intent['tag'] == tag:
            if tag == "next_task":
                return get_next_task()  # Respond with the next task
            else:
                return random.choice(intent['responses'])  # Return a random response from the matched intent

    return "Sorry, I didn't understand that."

def main():
    st.title("Task Management Chatbot")

    # Sidebar menu
    menu = ["Home", "Task History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the chatbot! Type your task, and I will manage it for you.")

        user_input = st.text_input("Enter task or ask for next task:")

        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120)

    elif choice == "Task History":
        st.write("Here are your stored tasks:")
        with open(TASK_FILE, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                st.write(f"Task: {row[0]}, Deadline: {row[1]}, Priority: {row[2]}")

    elif choice == "About":
        st.write("""
        This chatbot helps you manage your tasks and prioritize them based on deadlines.
        You can input your tasks along with deadlines, and it will keep track of them.
        When you ask, it will tell you your next task to do.
        """)

if __name__ == "__main__":
    main()

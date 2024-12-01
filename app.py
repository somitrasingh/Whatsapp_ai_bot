import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import openai
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import AgentExecutor, tool
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_functions
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field

# Load environment variables

# openai.api_key = os.environ['OPENAI_API_KEY']

# MongoDB Setup
uri = "mongodb+srv://somitrasinghkushwah:4wyNPmZEXewdKn09@cluster0.3jnci.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi("1"))
db = client.hotel_management
hotels_collection = db.hotels
bookings_collection = db.booking

# Define tools
class HotelDist(BaseModel):
    date: str = Field(..., description="Date for which hotels are searched")

@tool(args_schema=HotelDist)
def get_hotels(date):
    """
    Retrieve a list of hotels with availability on a specific date.
    
    :param date: Date string (e.g., "2024-11-23").
    :return: List of dictionaries containing hotel name, availability, and price.
    """
    query = {f"rooms.availability.{date}": {"$gt": 0}}
    projection = {"_id": 0, "hotel_name": 1, f"rooms.availability.{date}": 1, "price": 1}
    results = hotels_collection.find(query, projection)
    return [
        {
            "hotel_name": r["hotel_name"],
            "availability": r["rooms"]["availability"][date],
            "price": r["price"],
        }
        for r in results
    ]

@tool
def get_hotels_sorted_by_ratings() -> List:
    """
    Retrieve a list of hotels sorted by their ratings in descending order.
    Includes hotel name, ratings, availability, and price.
    :return: List of dictionaries containing hotel name, ratings, availability, and price.
    """
    projection = {
        "_id": 0,
        "hotel_name": 1,
        "ratings": 1,
        "rooms.availability": 1,
        "price": 1,
    }
    results = hotels_collection.find({}, projection).sort("ratings", -1)
    return list(results)

class HotelBook(BaseModel):
    hotel_name: str = Field(..., description="Name of hotel to be booked")
    date: str = Field(..., description="Date for which hotel is to booked")

@tool(args_schema=HotelBook)
def book_hotel(hotel_name, date) -> dict:
    """
    Books a hotel based on the input and updates the booking collection.
    
    :param hotel_name: Name of the hotel to book.
    :param date: Date for the booking (format: "YYYY-MM-DD").
    :return: Success message or error message if booking fails.
    """
    hotel = hotels_collection.find_one({"hotel_name": hotel_name, f"rooms.availability.{date}": {"$gt": 0}})
    if not hotel:
        return {"error": f"Hotel '{hotel_name}' not found or unavailable for date {date}."}
    update_result = hotels_collection.update_one(
        {"_id": hotel["_id"], f"rooms.availability.{date}": {"$gt": 0}},
        {"$inc": {f"rooms.availability.{date}": -1}},
    )
    if update_result.modified_count == 0:
        return {"error": "Failed to book the room. Availability might have changed."}
    booking_entry = {
        "hotel_id": hotel["_id"],
        "hotel_name": hotel_name,
        "date": date,
        "time": datetime.now().isoformat(),
        "price_per_night": hotel["price"],
    }
    bookings_collection.insert_one(booking_entry)
    return {"success": f"Room booked successfully at '{hotel_name}' for date {date}."}

tools = [get_hotels, get_hotels_sorted_by_ratings, book_hotel]
functions = [convert_to_openai_function(f) for f in tools]
model = ChatOpenAI(temperature=0).bind(functions=functions)

# Prompt setup
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI OrderBot specialized in assisting users with searching and booking hotels based on their preferences. Follow these guidelines strictly during interactions:

Greeting and Assistance: Start every conversation by greeting the user and asking how you can assist them. Avoid making assumptions until the user provides their query.

Understanding User Requests: Analyze the user's input to determine the most relevant action. Use the following functions as needed:

get_hotels(date): Use this to find hotels available for a specific date.
get_hotels_sorted_by_ratings(): Use this to provide a list of hotels ranked by their ratings.
book_hotel(hotel_name, date): Use this to book a hotel for the user once they specify the hotel name and date.
Gathering Required Details for Booking:

Before booking a hotel, ensure you collect all necessary details from the user:
Booking Date: Confirm the date for the booking.
Hotel Name: Ask the user to specify the hotel they wish to book.
Workflow:

Identify user intent accurately.
Invoke the appropriate function(s) for retrieving hotel options or making a booking.
Present information clearly and help users take the next step.
Ensure all required details for booking (hotel name and date) are confirmed before invoking the book_hotel function.
Fallback Behavior: If the user's query cannot be handled using any of the functions provided, use your general knowledge to answer or guide them appropriately. Always make it clear that you are offering an alternative solution.

Response Guidelines:

Be polite, clear, and concise.
Always confirm the user's preferences step by step when booking a hotel.
Use bullet points or structured formats for better readability when presenting options or results.
Example Interaction (FOR INSPIRATION ONLY – DO NOT COPY RESPONSES):

User: "I need a hotel for November 15th."
Assistant: "Let me find hotels available on November 15th for you. One moment!"
(Invoke get_hotels and return the options)

User: "Which hotels have the best ratings?"
Assistant: "Here are the hotels ranked by their ratings: [list of hotels]."
(Invoke get_hotels_sorted_by_rating and return the options)

User: "Book the Sunrise Hotel for November 15th."
Assistant: "Got it! Booking Sunrise Hotel for November 15th. Please hold on..."
(Invoke book_hotel and confirm the booking)
     Important Note: The examples provided are for reference only to illustrate how you might respond. You must always process queries logically and generate responses based on user input and current context."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
agent_chain = RunnablePassthrough.assign(
    agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
) | prompt | model | OpenAIFunctionsAgentOutputParser()

memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory)

# Flask App
app = Flask(__name__)

@app.route('/chatgpt', methods=['POST'])
def whatsapp_bot():
    user_input = request.values.get('Body', '').strip()
    response = MessagingResponse()
    try:
        result = agent_executor.invoke({"input": user_input})
        reply = result["output"]
    except Exception as e:
        reply = f"Error: {str(e)}"
    response.message(reply)
    return str(response)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)

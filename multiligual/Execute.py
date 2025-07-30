import argparse
import os
from dotenv import load_dotenv
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Multilingual Greeting App')
    parser.add_argument('--lang', 
                       choices=['en', 'hi', 'ta'], 
                       default='en',
                       help='Language code')
    
    args = parser.parse_args()
    
    # Load the appropriate language file
    env_file = f"msg_{args.lang}.env"
    
    if Path(env_file).exists(): load_dotenv(env_file, override=True)
    else:
        print(f"Language file not found, using English")
        load_dotenv("msg_en.env", override=True)
    
    # Get the greeting using the same key
    greeting = os.getenv("GREETINGS")
    print(f"{greeting}!")

if __name__ == "__main__":
    main()
from Intents.Intent_predictor import predict_intent

while True:
    text = input("Command > ")

    if text.lower() == "exit":
        break

    intent, conf = predict_intent(text)

    print("\nRESULT")
    print("Intent     :", intent)
    print("Confidence :", conf)


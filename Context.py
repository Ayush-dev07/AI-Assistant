class Context:
    def __init__(self):
        self.last_intent = None
        self.last_app = None
        self.last_value = None
        self.last_command = None

    def update(self, intent, entities, command):
        self.last_intent = intent
        self. last_command = command

        if "app" in entities:
            self.last_app = entities["app"]

        if "value" in entities:
            self.last_value = entities["value"]
        
    def resolve_pronouns(self, intent, entities, command):
        text = command.lower()

        if "it" in text or "that" in text:
            if intent in ["close_app", "open_app"] and "app" not in entities:
                if self.last_app:
                    entities["app"] = self.last_app
                
            if intent in ["set_volume", "set_brightness"] and "value" not in entities:
                if self.last_value:
                    entities["value"] = self.last_value
        return entities
"""Rule-based decision engine."""


class DecisionEngine:
    """Makes decisions based on screen information."""
    
    def __init__(self):
        self.rules = []
    
    def add_rule(self, condition, action):
        """Add a rule: if condition then action."""
        self.rules.append((condition, action))
    
    def evaluate(self, screen_data):
        """Evaluate rules against screen data."""
        pass

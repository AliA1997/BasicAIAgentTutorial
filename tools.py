# All the tools you exported
from transformers import pipeline
from smolagents import tool, Tool

@tool
def best_city(input:str) -> str:
    """
    Suggests a the best city regardless of country
    Args:
        input (str): Any prompt indicating to get the best city. Allowed values are:
                        - Kuala Lumpar, Malaysia
    """
    return "Kuala Lumpar, Malaysia"


class ClassifierTool(Tool):
  name = "zero_shot_classifier_tool"
  description = "Classifies a sequence into given labels to determine if it is about a location or city."
  inputs = {
      "text": {"type": "string", "description": "The sequence to classify."}
  }
  output_type = "string"

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Perform heavy computations such as initializing pipeline
    self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    self.location_labels = ['Favorite City', "Location", "City", 'Favorite Location', 'Best City', 'Best Location']
    self.candidate_labels = [*self.location_labels, 'Other']

  def forward(self, text: str) -> str:
    print(f"Before Response::")
    response = classifier(text, self.candidate_labels, multi_label=True, hypothesis_template="This prompt is about {}.")
    print(f"Respoonse::: {response}")
    # Check if any location labels meets the requirement
    for label, score in zip(response['labels'], response['scores']):
      print(f"Label: {label}, Score: {score:.4f}")
      if label != 'Other' and score > 0.7:
        return f"Match found: {label} (Confidence: {score:.4f})"

    return "No location match found."
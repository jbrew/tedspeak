import re

def lines_from_transcript(transcript):
	if isinstance(transcript, str):
		return handle_parens(transcript).split('. ')
	else:
		return []

def handle_parens(text):
	"""
	"(Laughter)" -> " <Laughter> "
	"(Applause)" -> " <Applause continues> " 
	"""
	return text.replace("(", ". <").replace(")", ">. ")
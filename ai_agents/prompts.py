"""The ONLY researcher-written text an agent ever sees. Review this file before running.

Everything else in an agent's conversation is read from the live task page at run time: the consent form, the instructions, the
practice/transition screens, the trial counter, the outcome ("You won $1" / "You lost $1"), "TOO SLOW!", error messages, and every
questionnaire item and response option. Nothing here mentions the study's hypotheses, the ITI manipulation or its purpose.

The two conditions differ in exactly one sentence (FIXATION_BLIND vs FIXATION_TOLD): whether the agent is told how long the
fixation cross between trials lasted. Humans experience that interval; a text-based agent cannot unless it is told.
"""

SYSTEM_PROMPT = (
    "You are taking part in an online research study as a participant. The study runs in a web browser. "
    "Each message shows you what is on the screen at that moment, as text and sometimes as an image of the screen. "
    "Respond to each screen in the format requested at the end of the message, and with nothing else. "
    "Take part as you normally would."
)

# --persona "..." appends one sentence to SYSTEM_PROMPT, e.g. --persona "You are 34 years old and were born in 1992."

# ---- how each kind of screen is presented --------------------------------------------------------------------------------------
SCREEN_HEADER = "[Screen]"
BUTTON_REPLY = "Reply with the label of the button you click: {buttons}."
TEXT_INPUT_REPLY = "Reply with exactly what you type into the box."
TRIAL_STIMULI = "Three images are shown side by side. The key shown under each image selects it: Q = left, T = middle, P = right."
TRIAL_REPLY = "Reply with the key you press: Q, T or P."
CHOICE_ECHO = "You pressed {key} ({position} image)."
FIXATION_BLIND = "The images disappear and a fixation cross (+) is shown."
FIXATION_TOLD = "The images disappear and a fixation cross (+) is shown for {duration}."   # "1 second" / "4 seconds"
FORM_ITEM_OPTIONS = "Options: {options}"
FORM_REPLY = ("Reply with one line per numbered statement, in the form <statement number>: <option number>, "
              "for all {n} statements.")
FORM_MISSING = "Please answer every statement. Missing or invalid: {missing}."
INVALID_REPLY = "That reply could not be used. {reply_format}"

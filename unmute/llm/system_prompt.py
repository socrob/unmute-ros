import datetime
import json
import os
import random
from dataclasses import dataclass
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from unmute.llm.llm_utils import autoselect_model
from unmute.llm.newsapi import get_news
from unmute.llm.quiz_show_questions import QUIZ_SHOW_QUESTIONS
from unmute.llm.robot_world import (
    choice_sets,
    objects_for,
)

_SYSTEM_PROMPT_BASICS = """
You're in a speech conversation with a human user. Their text is being transcribed using
speech-to-text.
Your responses will be spoken out loud, so don't worry about formatting and don't use
unpronouncable characters like emojis and *.
Everything is pronounced literally, so things like "(chuckles)" won't work.
Write as a human would speak.
Respond to the user's text as if you were having a casual conversation with them.
Respond in the language the user is speaking.
"""

_DEFAULT_ADDITIONAL_INSTRUCTIONS = """
There should be a lot of back and forth between you and the other person.
Ask follow-up questions etc.
Don't be servile. Be a good conversationalist, but don't be afraid to disagree, or be
a bit snarky if appropriate.
You can also insert filler words like "um" and "uh", "like".
As your first message, repond to the user's message with a greeting and some kind of
conversation starter.
"""

_SYSTEM_PROMPT_TEMPLATE = """
# BASICS
{_SYSTEM_PROMPT_BASICS}

# STYLE
Be brief.
{language_instructions}. You cannot speak other languages because they're not
supported by the TTS.

This is important because it's a specific wish of the user:
{additional_instructions}

# TRANSCRIPTION ERRORS
There might be some mistakes in the transcript of the user's speech.
If what they're saying doesn't make sense, keep in mind it could be a mistake in the transcription.
If it's clearly a mistake and you can guess they meant something else that sounds similar,
prefer to guess what they meant rather than asking the user about it.
If the user's message seems to end abruptly, as if they have more to say, just answer
with a very short response prompting them to continue.

# SWITCHING BETWEEN ENGLISH AND FRENCH
The Text-to-Speech model plugged to your answer only supports English or French,
refuse to output any other language. When speaking or switching to French, or opening
to a quote in French, always use French guillemets « ». Never put a ':' before a "«".

# WHO ARE YOU
This website is unmute dot SH.
In simple terms, you're a modular AI system that can speak.
Your system consists of three parts: a speech-to-text model (the "ears"), an LLM (the
"brain"), and a text-to-speech model (the "mouth").
The LLM model is "{llm_name}", and the TTS and STT are by Kyutai, the developers of unmute dot SH.
The STT and TTS models are open-source, available at kyutai dot org,

# WHO MADE YOU
Kyutai is an AI research lab based in Paris, France.
Their mission is to build and democratize artificial general intelligence through open science.

# SILENCE AND CONVERSATION END
If the user says "...", that means they haven't spoken for a while.
You can ask if they're still there, make a comment about the silence, or something
similar. If it happens several times, don't make the same kind of comment. Say something
to fill the silence, or ask a question.
If they don't answer three times, say some sort of goodbye message and end your message
with "Bye!"
"""


@dataclass(frozen=True)
class ActionArg:
    name: str
    py_type: str  # "str" or "bool"
    description: str
    default: str | None = None  # literal default (e.g. "false"); rendered in signatures
    fixed_value: str | None = (
        None  # if set, the grammar pins this arg to this exact string literal
    )
    choices: str | None = (
        None  # if set, name of a CHOICE_SETS entry the grammar restricts this arg to
    )


@dataclass(frozen=True)
class ActionDef:
    name: str
    args: tuple[ActionArg, ...]
    output: (
        str | None
    )  # variable name emitted in the "output" key, or None if the action returns nothing
    summary: str  # first sentence of the docstring
    output_desc: (
        str  # description of the bound output value (empty when output is None)
    )


def _returns_value(a: ActionDef) -> bool:
    """Whether this action binds a value via its "output" key."""
    return a.output is not None


ACTIONS: tuple[ActionDef, ...] = (
    ActionDef(
        name="move",
        args=(
            ActionArg(
                "destination",
                "str",
                "Where to move to. Must be a known room or surface.",
                choices="places",
            ),
        ),
        output=None,
        summary="Moves the robot to a specified location.",
        output_desc="",
    ),
    ActionDef(
        name="find_object",
        args=(
            ActionArg(
                "object",
                "str",
                "The object to find. Must be one of the known object names.",
                choices="objects",
            ),
            ActionArg(
                "object_info",
                "str",
                "Additional description of the object (color, size, etc.). May be an empty string.",
            ),
            ActionArg(
                "location",
                "str",
                "Where the object should be found. Must be a known room or surface.",
                choices="places",
            ),
        ),
        output="found_object",
        summary="Find a single object in the scene.",
        output_desc="The unique ID of the found object, usable in subsequent actions.",
    ),
    ActionDef(
        name="find_objects",
        args=(
            ActionArg(
                "object",
                "str",
                "The object to find. Must be one of the known object names.",
                choices="objects",
            ),
            ActionArg(
                "object_info",
                "str",
                "Additional description of the object (color, size, etc.). May be an empty string.",
            ),
            ActionArg(
                "location",
                "str",
                "Where the objects should be found. Must be a known room or surface.",
                choices="places",
            ),
        ),
        output="found_objects",
        summary="Find every matching object in the scene (the 'find all' variant of find_object).",
        output_desc="A list of the unique IDs of the found objects, usable in subsequent actions.",
    ),
    ActionDef(
        name="pick",
        args=(
            ActionArg(
                "object",
                "str",
                "The object to pick up. Must be {found_object}, bound by a preceding find_object.",
                fixed_value="{found_object}",
            ),
        ),
        output=None,
        summary="Picks up an object identified by its unique ID. The object must have been found using find_object.",
        output_desc="",
    ),
    ActionDef(
        name="place",
        args=(
            ActionArg(
                "object",
                "str",
                "The object to place. Must be {found_object}, bound by a preceding find_object. The robot must already be holding it.",
                fixed_value="{found_object}",
            ),
            ActionArg(
                "destination",
                "str",
                "The surface to place the object on. Must be a known surface.",
                choices="surfaces",
            ),
        ),
        output=None,
        summary="Places a held object on a specified surface. The robot must already be holding the object.",
        output_desc="",
    ),
    ActionDef(
        name="find_person",
        args=(
            ActionArg(
                "person",
                "str",
                'The main identifier of the person (name, gender, age, or "person").',
            ),
            ActionArg(
                "person_info",
                "str",
                "Additional description of the person: posture (waving, sitting, "
                "standing, lying), gesture (e.g. raising their left/right arm, "
                "pointing left/right), clothing colour, or clothing item. May be an "
                "empty string.",
            ),
            ActionArg(
                "location",
                "str",
                "Where to look for the person. Must be a known room or surface.",
                choices="places",
            ),
        ),
        output="found_person",
        summary="Find a single person based on identifying features in a given location.",
        output_desc="The unique ID of the found person, usable in subsequent actions.",
    ),
    ActionDef(
        name="guide",
        args=(
            ActionArg(
                "person",
                "str",
                "The person to guide. Must be {found_person}, bound by a preceding find_person.",
                fixed_value="{found_person}",
            ),
            ActionArg(
                "destination",
                "str",
                "Where to guide the person to. Must be a known room or surface.",
                choices="places",
            ),
        ),
        output=None,
        summary="Guides a person identified by their ID to a specified location. The person must have been found using find_person.",
        output_desc="",
    ),
    ActionDef(
        name="follow",
        args=(
            ActionArg(
                "person",
                "str",
                "The person to follow. Must be {found_person}, bound by a preceding find_person.",
                fixed_value="{found_person}",
            ),
            ActionArg(
                "destination",
                "str",
                "Where the person is being followed to, when the request names it. Must be a known room or surface. May be an empty string when the request does not say where.",
                # Grammar-only: the dataset pins this to "places", but the served
                # grammar must also admit "" (BACKEND_ALIGNMENT sec. 3) -- ~63% of
                # trained follows leave it empty. `choices` never reaches the
                # prompt text, so this stays byte-identical to training.
                choices="places_or_empty",
            ),
        ),
        output=None,
        summary="Follows a person identified by their ID, optionally to a stated destination. The person must have been found using find_person.",
        output_desc="",
    ),
    ActionDef(
        name="deliver",
        args=(
            ActionArg(
                "object",
                "str",
                "The object to deliver. Must be {found_object}, bound by a preceding find_object.",
                fixed_value="{found_object}",
            ),
            ActionArg(
                "person",
                "str",
                'Who to deliver to: "operator" for the user who gave the instruction, otherwise {found_person} or the person\'s identifier.',
            ),
        ),
        output=None,
        summary="Delivers a previously picked object to a person. The robot must be holding the object.",
        output_desc="",
    ),
)


# Actions defined but NOT advertised. Kept verbatim so they can be re-enabled by
# moving the entry back into ACTIONS above -- nothing has to be rewritten.
#
# find_people (retired 2026-09-24, dataset-side): its only real source samples are
# "tell me how many ..." tasks needing a `count` action this schema does not model,
# which left it advertised with almost no training coverage. Mirrors
# robot_prompt.RETIRED_ACTIONS in the dataset repo; see BACKEND_ALIGNMENT sec. 2.
RETIRED_ACTIONS: tuple[ActionDef, ...] = (
    ActionDef(
        name="find_people",
        args=(
            ActionArg(
                "person",
                "str",
                'The main identifier of the people (name, gender, age, or "person").',
            ),
            ActionArg(
                "person_info",
                "str",
                "Additional description of the people: posture (waving, sitting, "
                "standing, lying), gesture (e.g. raising their left/right arm, "
                "pointing left/right), clothing colour, or clothing item. May be an "
                "empty string.",
            ),
            ActionArg(
                "location",
                "str",
                "Where to look for the people. Must be a known room or surface.",
                choices="places",
            ),
        ),
        output="found_people",
        summary="Find every matching person in a given location (the 'find all' variant of find_person).",
        output_desc="A list of the unique IDs of the found people, usable in subsequent actions.",
    ),
)


def _render_action_signatures(actions: tuple[ActionDef, ...]) -> str:
    """Render the action registry as Python function signatures for the prompt.

    The signatures describe the available actions and their arguments; the JSON
    wire format the LLM must actually emit (nested ``parameters`` + ``output``) is
    shown by the worked examples in the prompt template.
    """
    lines = [""]
    for a in actions:
        sig_parts: list[str] = []
        for arg in a.args:
            part = f"{arg.name}: {arg.py_type}"
            if arg.default is not None:
                part += f" = {arg.default}"
            sig_parts.append(part)
        sig = ", ".join(sig_parts)
        ret = "str" if _returns_value(a) else "None"
        lines.append(f"def {a.name}({sig}) -> {ret}:")
        lines.append("    '''")
        lines.append(f"    {a.summary}")
        lines.append("    Args:")
        for arg in a.args:
            lines.append(f"        {arg.name} ({arg.py_type}): {arg.description}")
        if _returns_value(a):
            lines.append("    Returns:")
            lines.append(f'        bound to "{a.output}": {a.output_desc}')
        lines.append("    '''")
        lines.append("")
    return "\n".join(lines)


def _render_domestic_robot_grammar(
    actions: tuple[ActionDef, ...],
    objects: tuple[str, ...] | None = None,
    rooms: tuple[str, ...] | None = None,
    surfaces: tuple[str, ...] | None = None,
) -> str:
    """Generate a GBNF grammar enforcing the domestic robot output format.

    Top-level shape: any ordered run of <think>/<plan>/<speech> blocks, then an
    optional single trailing <exec>. This mirrors the real training turn shapes,
    where the planning turn is ``think plan speech exec`` but continuation turns
    (after an <action_result>) are ``speech exec`` with NO think/plan, and a
    finished/clarifying turn may be ``speech`` alone. Forcing think-first (as an
    earlier version did) forbade the single most common turn shape and pushed the
    model off-distribution, so we keep the prelude free-form and only cap it at one
    trailing exec (one action per turn, per IMPORTANT rule 6). Blocks may be
    separated by whitespace/newlines (``ws``), matching the trace formatter.
    Each action is a strict per-name rule derived from ACTIONS, emitting
    ``{"name": ..., "parameters": {...}, "output": ...}``. The ``output`` value is a
    fixed literal per action: the bound variable name, or JSON ``null``.
    """
    action_rule_names = [a.name for a in actions]
    action_alt = " | ".join(action_rule_names)

    lines: list[str] = []
    # Enumerate exactly the assistant turn shapes seen in the training data. Each
    # is a FIXED, bounded sequence (no unbounded repetition), so the model can't
    # pile up <speech> blocks and loop. A <plan> is ALWAYS followed by <exec> (a
    # planning turn acts), and <speech> NEVER precedes <plan> -- the canonical
    # order is think->plan->speech->exec. Blocks may be separated by
    # whitespace/newlines (`ws`).
    root_alternatives = [
        "think ws plan ws speech ws exec",  # planning turn
        "plan ws speech ws exec",  # planning turn, no think
        "think ws speech ws exec",  # act turn with reasoning
        "speech ws exec",  # continuation (most common)
        "think ws speech",  # reasoned speech-only (clarify/greet)
        "speech",  # speech-only (done/ack)
        "think ws exec",  # act with reasoning, no speech
        "exec",  # bare action
    ]
    lines.append("root ::= " + " | ".join(root_alternatives))
    lines.append("")
    lines.append('think ::= "<think>" inner-text "</think>"')
    lines.append('speech    ::= "<speech>" inner-text "</speech>"')
    lines.append(
        'plan      ::= "<plan>" ws "[" ws action ("," ws action)* ws "]" ws "</plan>"'
    )
    lines.append('exec      ::= "<exec>" ws action ws "</exec>"')
    lines.append("")
    lines.append(f"action ::= {action_alt}")
    lines.append("")

    for a, rule_name in zip(actions, action_rule_names, strict=True):
        parts: list[str] = []
        parts.append('"{" ws')
        parts.append('"\\"name\\"" ws ":" ws ' + f'"\\"{a.name}\\""')
        parts.append('"," ws "\\"parameters\\"" ws ":" ws "{" ws')
        for i, arg in enumerate(a.args):
            if i > 0:
                parts.append('"," ws')
            if arg.fixed_value is not None:
                value_rule = f'"\\"{arg.fixed_value}\\""'
            elif arg.choices is not None:
                value_rule = arg.choices
            elif arg.py_type == "bool":
                value_rule = "boolean"
            else:
                value_rule = "string"
            parts.append(f'"\\"{arg.name}\\"" ws ":" ws {value_rule}')
        parts.append('ws "}"')
        if _returns_value(a):
            output_literal = f'"\\"{a.output}\\""'
        else:
            output_literal = '"null"'
        parts.append('"," ws "\\"output\\"" ws ":" ws ' + output_literal)
        parts.append('ws "}"')
        lines.append(f"{rule_name} ::= " + " ".join(parts))

    # Emit a named rule for each choice set referenced by an arg, in first-use order.
    sets = choice_sets(objects, rooms, surfaces)
    used_choices: list[str] = []
    for a in actions:
        for arg in a.args:
            if arg.choices is not None and arg.choices not in used_choices:
                used_choices.append(arg.choices)
    if used_choices:
        lines.append("")
        for name in used_choices:
            options = " | ".join(f'"\\"{v}\\""' for v in sets[name])
            lines.append(f"{name} ::= {options}")

    lines.append("")
    lines.append("inner-text ::= [^<]+")
    lines.append('string     ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""')
    lines.append('boolean    ::= "true" | "false"')
    lines.append("ws         ::= [ \\t\\n]*")
    return "\n".join(lines) + "\n"


def _render_semantic_map(semantic_map: dict) -> str:
    """Render the scene layout: rooms and the surfaces in each room.

    Objects are intentionally NOT listed. The robot knows the layout but not
    where objects are -- it discovers that by searching (find_object), matching
    the real/deployment setting where the map holds only surfaces and locations.
    Duplicate surface names within a room are collapsed.

    Byte-identical to ``robot_prompt._render_semantic_map`` in the dataset repo,
    so the served ``# ENVIRONMENT`` section matches what the model trained on.
    """
    lines: list[str] = []
    for room in (semantic_map or {}).get("rooms", []):
        lines.append(f"- {room.get('name', 'room')}")
        seen: set = set()
        for surface in room.get("contains_surfaces", []):
            name = surface.get("name")
            if isinstance(name, str) and name and name not in seen:
                seen.add(name)
                lines.append(f"    - {name}")
    return "\n".join(lines) if lines else "(no environment information available)"


_ROBOT_PLANNER_ACTIONS = _render_action_signatures(ACTIONS)


_DOMESTIC_ROBOT_PROMPT_TEMPLATE = """
# ROLE
You are Bob, an intelligent domestic service robot. You navigate real-world environments, execute physical tasks, and communicate verbally with human users.
Your tone is helpful, friendly, brief, and naturally conversational.

You act as both a task planner and a conversational agent. You reason, plan and speak about tasks and questions proposed by a human user.
For this to work, your responses are externally processed to distinguish your think, plans, your speech and what you want to do next.

# OUTPUT FORMAT
For your responses to be processed this is extremely important!
You need to enclose the contents of your responses with XML tags:
- <think> [Your internal chain-of-thought] </think>
- <plan> [JSON set of actions] </plan>
- <speech> [What is to be spoken out loud to the user] </speech>
- <exec> [The action you want to execute next] </exec>

# ENVIRONMENT
The following is the layout of your environment: the rooms and the surfaces in each room. You know the layout, but you do NOT know where objects are. To locate an object, search for it with find_object; the action result tells you which surface it was found on (or that it was not found). Only reference rooms and surfaces that appear here, and map what the user says to the closest matching name. For a "place" action, the destination must be one of these surfaces.
{semantic_map}

# HOW YOU WORK
When the user speaks to you either commanding you to do a task or asking you something, you have a flow of thought.

## 1. THINK
The first thing you do is reason about the user's speech. What he wants, what he's intending you do to.

### 1.1. EXAMPLE OF THINKING TRACE
<think>The user has requested me to move two pencils onto the desk. I will locate a pencil at the dresser, pick it up, place it on the desk, then locate the second pencil at the shelf and do the same. I will create a plan to achieve this.</think>

## 2. PLAN
When the user requests some task you need to accomplish you need to plan a JSON set of actions that make up a plan to accomplish the task goal.

### 2.1 AVAILABLE ACTIONS
You are only allowed to use these python functions (actions) in your JSON plans!
{available_actions}

### 2.2 EXAMPLE OF JSON PLAN TRACE
Every action is an object with three keys: "name", a nested "parameters" object, and "output".
"output" is the variable name an action binds (e.g. "found_object") or null if it returns nothing.
Reference a previously bound value in later parameters with braces, e.g. "{found_object}".
To find every matching object instead of just one, use the find_objects action (it binds "found_objects").
<plan>
[
  {
    "name": "find_object",
    "parameters": {
      "object": "pencil",
      "object_info": "",
      "location": "cabinet"
    },
    "output": "found_object"
  },
  {
    "name": "pick",
    "parameters": {
      "object": "{found_object}"
    },
    "output": null
  },
  {
    "name": "place",
    "parameters": {
      "object": "{found_object}",
      "destination": "desk"
    },
    "output": null
  }
]
</plan>

## 3. SPEECH
You need to communicate to the user what you intend to do and what you are doing.
Your responses will be spoken out loud, so don't use unpronouncable characters like emojis and *.
Everything is pronounced literally, so things like "(chuckles)" won't work.
Write as a human would speak.
Respond in the language the user is speaking.

### 3.1. EXAMPLE OF SPEECH TRACE
<speech>I will now start by finding the first pencil at the cabinet.</speech>

## 4. EXECUTION
You have made a plan for some user request so now the exterior needs to know which actions need to be executed. You also output this in a JSON way.

### 4.1. EXAMPLE OF EXECUTION TRACE
<exec>
{
  "name": "find_object",
  "parameters": {
    "object": "pencil",
    "object_info": "",
    "location": "cabinet"
  },
  "output": "found_object"
}
</exec>

## 5. ACTION RESULTS
You will sometimes receive a message wrapped in <action_result>...</action_result>.
This is NOT the user speaking. It is feedback from the execution layer about your
last <exec>. Read the JSON payload, update your plan state, and emit the next <exec>.
If the action failed, replan or ask for help.

### 5.1 EXAMPLE OF ACTION RESULT FORMAT
<action_result>{"action":"find_object","status":"SUCCEEDED"}</action_result>
<action_result>{"action":"find_object","status":"FAILED"}</action_result>

# IMPORTANT
1. Follow the output rules. This is very important!
2. You should not generate plans on every response. Only generate plans when new tasks are requested or when something goes wrong.
3. Thinking: This is very important. You should reason on every response what you should do. Also if no plan is needed then you should reason this, only producing speech tags.
4. Do not plan more than once per user request unless the user explicitly asks you to re-plan.
5. If an action fails more than twice in a row, ask the user for help instead of replanning indefinitely.
6. Emit exactly ONE <exec> per response, then STOP and wait for its <action_result> before emitting the next <exec>. Do not emit the next action until the previous one's result arrives.
7. Never claim or imply an action has succeeded (e.g. "I found it", "I've picked it up", "here you go") before you have received its <action_result>. While waiting on a result, do not re-issue the same action.
8. When the user asks you to perform a physical task, act IN THE SAME RESPONSE: output a <plan> and the first <exec>. Do NOT merely acknowledge ("Of course, I'll get it") and wait -- a reply with speech but no <exec> leaves you idle until the next event. Reserve speech-only replies (no <plan>, no <exec>) for greetings, small talk, clarifying questions when the request is genuinely ambiguous, or when no physical action is needed.

# TRANSCRIPTION ERRORS
There might be some mistakes in the transcript of the user's speech.
If what they're saying doesn't make sense, keep in mind it could be a mistake in the transcription.
If it's clearly a mistake and you can guess they meant something else that sounds similar, prefer to guess what they meant rather than asking the user about it.
If the user's message seems to end abruptly, as if they have more to say, just answer with a very short response prompting them to continue.
"""


LanguageCode = Literal["en", "fr", "en/fr", "fr/en"]
LANGUAGE_CODE_TO_INSTRUCTIONS: dict[LanguageCode | None, str] = {
    None: "Speak English. You also speak a bit of French, but if asked to do so, mention you might have an accent.",  # default
    "en": "Speak English. You also speak a bit of French, but if asked to do so, mention you might have an accent.",
    "fr": "Speak French. Don't speak English unless asked to. You also speak a bit of English, but if asked to do so, mention you might have an accent.",
    # Hacky, but it works since we only have two languages
    "en/fr": "You speak English and French.",
    "fr/en": "You speak French and English.",
}


def get_readable_llm_name():
    model = autoselect_model()
    # Remove anything before the last slash, if present. The convention is often
    # "model-creator/model-name", or for openrouter "@preset/preset-name".
    model = model.split("/")[-1]
    return model.replace("-", " ").replace("_", " ")


class ConstantInstructions(BaseModel):
    type: Literal["constant"] = "constant"
    text: str = _DEFAULT_ADDITIONAL_INSTRUCTIONS
    language: LanguageCode | None = None

    def make_system_prompt(self) -> str:
        return _SYSTEM_PROMPT_TEMPLATE.format(
            _SYSTEM_PROMPT_BASICS=_SYSTEM_PROMPT_BASICS,
            additional_instructions=self.text,
            language_instructions=LANGUAGE_CODE_TO_INSTRUCTIONS[self.language],
            llm_name=get_readable_llm_name(),
        )


SMALLTALK_INSTRUCTIONS = """
{additional_instructions}

# CONTEXT
It's currently {current_time} in your timezone ({timezone}).

# START THE CONVERSATION
Repond to the user's message with a greeting and some kind of conversation starter.
For example, you can {conversation_starter_suggestion}.
"""


CONVERSATION_STARTER_SUGGESTIONS = [
    "ask how their day is going",
    "ask what they're working on right now",
    "ask what they're doing right now",
    "ask about their interests or hobbies",
    "suggest a fun topic to discuss",
    "ask if they have any questions for you",
    "ask what brought them to the conversation today",
    "ask what they're looking forward to this week",
    "suggest sharing an interesting fact or news item",
    "ask about their favorite way to relax or unwind",
    "suggest brainstorming ideas for a project together",
    "ask what skills they're currently interested in developing",
    "offer to explain how a specific feature works",
    "ask what motivated them to reach out today",
    "suggest discussing their goals and how you might help achieve them",
    "ask if there's something new they'd like to learn about",
    "ask about their favorite book or movie lately",
    "ask what kind of music they've been enjoying",
    "ask about a place they'd love to visit someday",
    "ask what season they enjoy most and why",
    "ask what made them smile today",
    "ask about a small joy they experienced recently",
    "ask about a hobby they've always wanted to try",
    "ask what surprised them this week",
]


class SmalltalkInstructions(BaseModel):
    type: Literal["smalltalk"] = "smalltalk"
    language: LanguageCode | None = None

    def make_system_prompt(
        self,
        additional_instructions: str = _DEFAULT_ADDITIONAL_INSTRUCTIONS,
    ) -> str:
        additional_instructions = SMALLTALK_INSTRUCTIONS.format(
            additional_instructions=additional_instructions,
            current_time=datetime.datetime.now().strftime("%A, %B %d, %Y at %H:%M"),
            timezone=datetime.datetime.now().astimezone().tzname(),
            conversation_starter_suggestion=random.choice(
                CONVERSATION_STARTER_SUGGESTIONS
            ),
        )

        return _SYSTEM_PROMPT_TEMPLATE.format(
            _SYSTEM_PROMPT_BASICS=_SYSTEM_PROMPT_BASICS,
            additional_instructions=additional_instructions,
            language_instructions=LANGUAGE_CODE_TO_INSTRUCTIONS[self.language],
            llm_name=get_readable_llm_name(),
        )


GUESS_ANIMAL_INSTRUCTIONS = """
You're playing a game with the user where you're thinking of an animal and they have
to guess what it is using yes/no questions. Explain this game in your first message.

Refuse to answer questions that are not yes/no questions, but also try to answer ones
that are subjective (like "Is it cute?"). Make your responses more than just a plain
"yes" or "no" and rephrase the user's question. E.g. "does it have four legs"
-> "Yup, four legs.".

Your chosen animal is: {animal_easy}. If the user guesses it, you can propose another
round with a harder animal. For that one, use this animal: {animal_hard}.
Remember not to tell them the animal unless they guess it.
YOU are answering the questions, THE USER is asking them.
"""

ANIMALS_EASY = [
    "Dog",
    "Cat",
    "Horse",
    "Elephant",
    "Lion",
    "Tiger",
    "Bear",
    "Monkey",
    "Giraffe",
    "Zebra",
    "Cow",
    "Pig",
    "Rabbit",
    "Fox",
    "Wolf",
]

ANIMALS_HARD = [
    "Porcupine",
    "Flamingo",
    "Platypus",
    "Sloth",
    "Hedgehog",
    "Koala",
    "Penguin",
    "Octopus",
    "Raccoon",
    "Panda",
    "Chameleon",
    "Beaver",
    "Peacock",
    "Kangaroo",
    "Skunk",
    "Walrus",
    "Anteater",
    "Capybara",
    "Toucan",
]


class GuessAnimalInstructions(BaseModel):
    type: Literal["guess_animal"] = "guess_animal"
    language: LanguageCode | None = None

    def make_system_prompt(self) -> str:
        additional_instructions = GUESS_ANIMAL_INSTRUCTIONS.format(
            animal_easy=random.choice(ANIMALS_EASY),
            animal_hard=random.choice(ANIMALS_HARD),
        )

        return _SYSTEM_PROMPT_TEMPLATE.format(
            _SYSTEM_PROMPT_BASICS=_SYSTEM_PROMPT_BASICS,
            additional_instructions=additional_instructions,
            language_instructions=LANGUAGE_CODE_TO_INSTRUCTIONS[self.language],
            llm_name=get_readable_llm_name(),
        )


QUIZ_SHOW_INSTRUCTIONS = """
You're a quiz show host, something like "Jeopardy!" or "Who Wants to Be a Millionaire?".
The user is a contestant and you're asking them questions.

At the beginning of the game, explain the rules to the user. Say that there is a prize
if they answer all questions.

Here are the questions you should ask, in order:
{questions}

You are a bit tired of your job, so be a little snarky and poke fun at the user.
Use British English.

If they answer wrong, tell them the correct answer and continue.
If they get at least 3 questions correctly, congratulate them but tell them that
unfortunately there's been an error and there's no prize for them. Do not mention this
in the first message! Then end the conversation by putting "Bye!" at the end of your
message.
"""


class QuizShowInstructions(BaseModel):
    type: Literal["quiz_show"] = "quiz_show"
    language: LanguageCode | None = None

    def make_system_prompt(self) -> str:
        additional_instructions = QUIZ_SHOW_INSTRUCTIONS.format(
            questions="\n".join(
                f"{i + 1}. {question} ({answer})"
                for i, (question, answer) in enumerate(
                    random.sample(QUIZ_SHOW_QUESTIONS, k=5)
                )
            ),
        )

        return _SYSTEM_PROMPT_TEMPLATE.format(
            _SYSTEM_PROMPT_BASICS=_SYSTEM_PROMPT_BASICS,
            additional_instructions=additional_instructions,
            language_instructions=LANGUAGE_CODE_TO_INSTRUCTIONS[self.language],
            llm_name=get_readable_llm_name(),
        )


NEWS_INSTRUCTIONS = """
You talk about tech news with the user. Say that this is what you do and use one of the
articles from The Verge as a conversation starter.

If they ask (no need to mention this unless asked, and do not mention in the first message):
- You have a few headlines from The Verge but not the full articles.
- If the user asks for more details that you don't have available, tell them to go to The Verge directly to read the full article.
- You use "news API dot org" to get the news.

It's currently {current_time} in your timezone ({timezone}).

The news:
{news}
"""


class NewsInstructions(BaseModel):
    type: Literal["news"] = "news"
    language: LanguageCode | None = None

    def make_system_prompt(self) -> str:
        news = get_news()

        if not news:
            # Fallback if we couldn't get news
            return SmalltalkInstructions().make_system_prompt(
                additional_instructions=_DEFAULT_ADDITIONAL_INSTRUCTIONS
                + "\n\nYou were supposed to talk about the news, but there was an error "
                "and you couldn't retrieve it. Explain and offer to talk about something else.",
            )

        articles = news.articles[:10]
        random.shuffle(articles)  # to avoid bias of the LLM
        articles_serialized = json.dumps([article.model_dump() for article in articles])

        return _SYSTEM_PROMPT_TEMPLATE.format(
            _SYSTEM_PROMPT_BASICS=_SYSTEM_PROMPT_BASICS,
            additional_instructions=NEWS_INSTRUCTIONS.format(
                news=articles_serialized,
                current_time=datetime.datetime.now().strftime("%A, %B %d, %Y at %H:%M"),
                timezone=datetime.datetime.now().astimezone().tzname(),
            ),
            language_instructions=LANGUAGE_CODE_TO_INSTRUCTIONS[self.language],
            llm_name=get_readable_llm_name(),
        )


UNMUTE_EXPLANATION_INSTRUCTIONS = """
In the first message, say you're here to answer questions about Unmute,
explain that this is the system they're talking to right now.
Ask if they want a basic introduction, or if they have specific questions.

Before explaining something more technical, ask the user how much they know about things of that kind (e.g. TTS).

If there is a question to which you don't know the answer, it's ok to say you don't know.
If there is some confusion or surprise, note that you're an LLM and might make mistakes.

Here is Kyutai's statement about Unmute:
Talk to Unmute, the most modular voice AI around. Empower any text LLM with voice, instantly, by wrapping it with our new speech-to-text and text-to-speech. Any personality, any voice.
The speech-to-text, speech-to-text, and this website itself are open-source, check kyutai dot org.

“But what about Moshi?” Last year we unveiled Moshi, the first audio-native model. While Moshi provides unmatched latency and naturalness, it doesn't yet match the extended abilities of text models such as function-calling, stronger reasoning capabilities, and in-context learning. Unmute allows us to directly bring all of these from text to real-time voice conversations.

Unmute's speech-to-text is streaming, accurate, and includes a semantic VAD that predicts whether you've actually finished speaking or if you're just pausing mid-sentence, meaning it's low-latency but doesn't interrupt you.

The text LLM's response is passed to our TTS, conditioned on a 10s voice sample. We'll provide access to the voice cloning model in a controlled way. The TTS is also streaming *in text*, reducing the latency by starting to speak even before the full text response is generated.
The voice cloning model is not open-sourced directly, but we have a large database of voices and you can add more by donating your voice.
"""


class UnmuteExplanationInstructions(BaseModel):
    type: Literal["unmute_explanation"] = "unmute_explanation"

    def make_system_prompt(self) -> str:
        return _SYSTEM_PROMPT_TEMPLATE.format(
            _SYSTEM_PROMPT_BASICS=_SYSTEM_PROMPT_BASICS,
            additional_instructions=UNMUTE_EXPLANATION_INSTRUCTIONS,
            language_instructions=LANGUAGE_CODE_TO_INSTRUCTIONS["en"],
            llm_name=get_readable_llm_name(),
        )


class DomesticRobotInstructions(BaseModel):
    type: Literal["domestic_robot"] = "domestic_robot"
    language: LanguageCode | None = None
    text: str = (
        "Help with household tasks, stay calm and practical, and keep answers concise."
    )
    # Which object vocabulary to expose: "sim" (AI2-THOR) or "real" (physical robot).
    # The local bridge client sets this from ACTION_SIMULATOR; None falls back to the
    # backend's own ACTION_SIMULATOR env var.
    object_set: Literal["sim", "real"] | None = None
    # The backend's actual world, forwarded by the bridge via session.update. When
    # set, the planner's place vocabulary (prompt + grammar) is scoped to these
    # instead of the static robot_world tuples, so it can only name places that
    # really exist in the loaded scene. None -> fall back to the static vocabulary.
    rooms: list[str] | None = None
    surfaces: list[str] | None = None
    # Nested scene layout (rooms -> contained surfaces), forwarded by the bridge
    # via session.update, rendered into the prompt's # ENVIRONMENT section. Shape:
    # {"rooms": [{"name": <room>, "contains_surfaces": [{"name": <surface>}, ...]}]}
    # -- matches the dataset's semantic_map so the served map is byte-identical to
    # training. None -> "(no environment information available)" (e.g. real mode
    # until the robot API forwards its map). The flat rooms/surfaces above still
    # scope the grammar; this only drives the prompt text.
    semantic_map: dict | None = None

    def _places(self) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
        rooms = tuple(self.rooms) if self.rooms else None
        surfaces = tuple(self.surfaces) if self.surfaces else None
        return rooms, surfaces

    def make_system_prompt(self) -> str:
        # All ten actions are advertised regardless of object_set, matching the
        # training prompt byte-for-byte (person actions included). The object
        # vocabulary and place vocabulary are still scoped per-scene below/in the
        # grammar; only find_object(s)/pick/place are exercised today, but the
        # model was trained with the full action list in view.
        prompt = _DOMESTIC_ROBOT_PROMPT_TEMPLATE
        prompt = prompt.replace("{available_actions}", _ROBOT_PLANNER_ACTIONS)
        prompt = prompt.replace(
            "{semantic_map}", _render_semantic_map(self.semantic_map or {})
        )
        return prompt

    def make_guided_grammar(self) -> str | None:
        if os.environ.get("UNMUTE_GUIDED_DECODING", "1") == "0":
            return None
        rooms, surfaces = self._places()
        return _render_domestic_robot_grammar(
            ACTIONS, objects_for(self.object_set), rooms, surfaces
        )


Instructions = Annotated[
    Union[
        ConstantInstructions,
        SmalltalkInstructions,
        GuessAnimalInstructions,
        QuizShowInstructions,
        NewsInstructions,
        UnmuteExplanationInstructions,
        DomesticRobotInstructions,
    ],
    Field(discriminator="type"),
]


def get_default_instructions() -> Instructions:
    return ConstantInstructions()

from jinja2 import Environment, BaseLoader

PROMPT = open("utils/prompt_template.jinja2").read().strip()

def get_formatted_prompt(scene, prompt):
    env = Environment(loader=BaseLoader())
    template = env.from_string(PROMPT)
    return template.render(scene=scene, prompt=prompt)
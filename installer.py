from framework.utils import load_use_cases
from framework.tkinker_ui import UI

data = load_use_cases()

UI(data).run()

import pdb
from enum import Enum

"""
    Helper module to print splitted sections and paragraphs in color
    Allows to visually check if the detected sections and paragraphs are correct

    Methods:
        `visualize_sections(sections: dict, compact: bool = True)`
        `visualize_paragraphs(sections: dict)`
        `visualize_sections_and_break(sections: dict, compact: bool = False)`
        `visualize_paragraphs_and_break(sections: dict)`

    Usage:
        include with `import scrc.utils.debug_utils as debug`
        call with `debug.<function>(list_of_sections, optional: compact)`
        optional: pass `compact=True` to print the sections in a compact way
        the `list_of_sections` should be in the standard section_splitting output format, e.g.
        `{'section_name': [list_of_paragraphs]}`

"""

# ANSI codes for colors in terminal
class Color(Enum):
    BLACK = '\u001b[30m'
    RED = '\u001b[31m'
    GREEN = '\u001b[32m'
    YELLOW = '\u001b[33m'
    BLUE = '\u001b[34m'
    MAGENTA = '\u001b[35m'
    CYAN = '\u001b[36m'
    WHITE = '\u001b[37m'
    RESET = '\u001b[0m'

def visualize_paragraphs(sections: dict):
    res = ''
    colors = [Color.BLUE.value, Color.GREEN.value]
    for section in sections:
        res += Color.YELLOW.value + section.name + ' ---------------- \n' + Color.RESET.value
        for index, paragraphs in enumerate(sections[section]):
            import pdb; pdb.set_trace()
            res += colors[index % 2]
            res += ''.join(paragraphs)
        res += '\n\n' + Color.RESET.value

    res += Color.RESET.value
    print(res)


def visualize_sections(sections: dict, compact: bool = True):
    color_list = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.BLUE, Color.MAGENTA, Color.CYAN, Color.WHITE]
    if len(sections) > len(color_list):
        color_list = int(len(sections) / len(color_list)) * color_list
  
    sections_texts = [color.value + ''.join(paragraphs) for color, paragraphs in zip(color_list, sections.values())]
    res = ''
    if compact:
        for section in sections_texts:
          if len(section) > 100:
            res += section[:50] + ' < ... > ' + section[-50:]
          else:
            res += section
    else:
        res = ''.join(sections_texts)

    res += Color.RESET.value
    print(res)

def visualize_sections_and_break(sections: dict, compact: bool = False):
    visualize_sections(sections, compact)
    pdb.set_trace()

def visualize_paragraphs_and_break(sections: dict):
    visualize_paragraphs(sections)
    pdb.set_trace()



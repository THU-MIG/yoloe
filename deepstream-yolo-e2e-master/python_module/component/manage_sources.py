"""
Creative Commons Attribution-NonCommercial 4.0 International License

You are free to share and adapt the material under the following terms:
- Attribution: Give appropriate credit.
- NonCommercial: Not for commercial use without permission.

For inquiries: levi.pereira@gmail.com
Repository: DeepStream / YOLO (https://github.com/levipereira/deepstream-yolo-e2e)
License: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""

import configparser
import os
from prettytable import PrettyTable
import sys
from python_module.common.utils import display_message


# Load the config.ini file
config_file = "config/python_app/media.ini"
config = configparser.ConfigParser()

if os.path.exists(config_file):
    config.read(config_file)
else:
    display_message("e", f"Configuration File {config_file} not found.")
    sys.exit(1)

def save_config():
    with open(config_file, 'w') as configfile:
        config.write(configfile)
    display_message("s","Changes saved.")

def sanitize_input(text):
    return text.replace('"', '').replace("'", "").strip()

def get_active_sources():
    if os.path.exists(config_file):
        config.read(config_file)
        num_active_sources = sum(1 for section in config.sections() if config[section]['enable'] == '1')
    return num_active_sources


def validate_url(media_type, url):
    if media_type == 'rtsp':
        if not url.startswith('rtsp://'):
            return False, "RTSP URL must start with 'rtsp://'."
    elif media_type == 'file':
        if not os.path.isfile(url):
            return False, f"The file '{url}' does not exist."
    elif media_type == 'youtube':
        if not url.startswith('https://'):
            return False, "YouTube URL must start with 'https://'."
    elif media_type in ['http', 'https']:
        if not (url.startswith('http://') or url.startswith('https://')):
            return False, "HTTP/HTTPS URL must start with 'http://' or 'https://'."
    
    return True, ""

def add_media():
    media_types = {
        '1': 'youtube',
        '2': 'rtsp',
        '3': 'file',
        '4': 'http',
        '5': 'https'
    }

    display_message("d","Choose media type:")
    for key, value in media_types.items():
        display_message("d",f"{key}. {value}")

    media_type_choice = input("Enter the number of the media type: ").strip()
    media_type = media_types.get(media_type_choice)

    if not media_type:
        display_message("e","Invalid media type selected.")
        return

    media_name = sanitize_input(input("Enter media name: ").strip())
    url = input("Enter the URL: ").strip()
    
    # Validate the URL based on the selected media type
    is_valid, message = validate_url(media_type, url)
    if not is_valid:
        display_message("e",message)
        return

    enable = input("Enable this media? (yes/no): ").strip().lower()
    enable_value = '1' if enable in ['yes', 'y'] else '0'
    
    # Find the next available index without exposing the Media ID
    index = 0
    while f'MediaSettings-{index}' in config:
        index += 1
    
    section = f'MediaSettings-{index}'
    config[section] = {
        'media_name': media_name,
        'type': media_type,
        'url': url,
        'enable': enable_value
    }
    display_message("s",f"Added {media_name} with type {media_type} and URL {url}.")
    save_config()

def remove_media():
    display_message("d","\nAvailable Media Sources:")
    table = PrettyTable()
    table.field_names = ["Index", "Media Name", "Media Type", "URL"]
    
    # Set alignment: center index, left-align other columns
    table.align["Index"] = "c"  # Center align index
    table.align["Media Name"] = "l"  # Left align media name
    table.align["Media Type"] = "l"  # Left align media type
    table.align["URL"] = "l"  # Left align URL
    
    media_list = []  # Store media sections for accurate indexing

    for count, section in enumerate(config.sections(), start=1):
        media_name = sanitize_input(config[section]['media_name'])
        media_type = config[section]['type']
        url = config[section]['url']
        table.add_row([count, media_name, media_type, url])
        media_list.append(section)  # Keep track of the sections
    
    display_message("d",table)
    
    choice = input("Enter the user-friendly index to remove (or 'c' to cancel): ").strip()
    if choice.lower() == 'c':
        display_message("d","Removal cancelled.")
        return
    
    try:
        index = int(choice) - 1  # Convert to zero-based index
        if 0 <= index < len(media_list):
            section = media_list[index]  # Use the tracked section for removal
            media_name = config[section]['media_name']
            config.remove_section(section)
            display_message("d",f"{media_name} removed.")
            save_config()
        else:
            display_message("e",f"Media with index {index + 1} not found.")
    except (ValueError, IndexError):
        display_message("e","Invalid index format. Please enter a valid number.")

def list_summary():
    active_count = sum(1 for section in config.sections() if config[section]['enable'] == '1')
    inactive_count = sum(1 for section in config.sections() if config[section]['enable'] == '0')

    summary_table = PrettyTable()
    summary_table.field_names = ["Status", "Count"]
    summary_table.add_row(["Active", active_count])
    summary_table.add_row(["Inactive", inactive_count])
    display_message("d", "Media Summary:")
    display_message("d", summary_table)

def list_media():
    display_message("d","\nAll Media Sources:")
    table = PrettyTable()
    table.field_names = [ "Media Name", "Media Type", "URL", "Status"]
    
    # Set alignment: center index, left-align other columns
    #table.align["Index"] = "c"  # Center align index
    table.align["Media Name"] = "l"  # Left align media name
    table.align["Media Type"] = "l"  # Left align media type
    table.align["URL"] = "l"  # Left align URL
    table.align["Status"] = "l"  # Left align status
    
    for count, section in enumerate(config.sections(), start=1):
        media_name = sanitize_input(config[section]['media_name'])
        media_type = config[section]['type']
        url = config[section]['url']
        status = "Enabled" if config[section]['enable'] == '1' else "Disabled"
        table.add_row([media_name, media_type, url, status])
    
    display_message("d", table)

def list_active_media():
    display_message("d","\nActive Media Sources:")
    table = PrettyTable()
    table.field_names = ["Media Name", "Media Type", "URL", "Status"]
    
    # Set alignment: left-align columns
    table.align["Media Name"] = "l"  # Left align media name
    table.align["Media Type"] = "l"  # Left align media type
    table.align["URL"] = "l"  # Left align URL
    table.align["Status"] = "l"  # Left align status
    
    active_found = False  # Flag to check if any active media is found

    for count, section in enumerate(config.sections(), start=1):
        status = "Enabled" if config[section]['enable'] == '1' else "Disabled"
        
        # Only add the row if the media is enabled
        if status == "Enabled":
            active_found = True
            media_name = sanitize_input(config[section]['media_name'])
            media_type = config[section]['type']
            url = config[section]['url']
            table.add_row([media_name, media_type, url, status])
    
    if active_found:
        display_message("d",table)
        return True
    else:
        display_message("w","No active media sources found.")
        return False

def activate_media():
    display_message("d","\nInactive Media Sources:")
    inactive_medias = [(section, config[section]['media_name']) for section in config.sections() if config[section]['enable'] == '0']
    
    if not inactive_medias:
        display_message("w","No inactive media sources available.")
        return

    table = PrettyTable()
    table.field_names = ["Index", "Media Name", "Media Type", "URL"]
    
    # Set alignment: center index, left-align other columns
    table.align["Index"] = "c"  # Center align index
    table.align["Media Name"] = "l"  # Left align media name
    table.align["Media Type"] = "l"  # Left align media type
    table.align["URL"] = "l"  # Left align URL
    
    for i, (section, media_name) in enumerate(inactive_medias, 1):
        table.add_row([i, sanitize_input(media_name), config[section]['type'], config[section]['url']])
    
    display_message("d", table)
    
    choice = input("Enter the numbers of the media to activate (e.g., '1,2,3') or 'c' to cancel: ").strip()
    if choice.lower() == 'c':
        display_message("d","Activation cancelled.")
        return
    
    try:
        # Convert the user input into a list of indices, handling both single and multiple inputs
        indices = [int(x.strip()) - 1 for x in choice.split(',') if x.strip().isdigit()]
        
        if not indices:
            raise ValueError("No valid selections.")
        
        # Remove duplicates from the input
        indices = list(set(indices))

        # Check for valid index range
        invalid_indices = [i for i in indices if i < 0 or i >= len(inactive_medias)]
        if invalid_indices:
            raise IndexError(f"Invalid selection(s): {', '.join(map(str, [i + 1 for i in invalid_indices]))}. Please try again.")
        
        # Activate the selected media sources
        activated_medias = []
        for index in indices:
            section, media_name = inactive_medias[index]
            config[section]['enable'] = '1'
            activated_medias.append(media_name)
        
        # Display success message for activated media
        display_message("s", f"The following media sources have been activated: {', '.join(activated_medias)}.")
        save_config()
    
    except (IndexError, ValueError) as e:
        display_message("e", str(e))
        display_message("e", "Invalid input. Please enter valid numbers separated by commas.")




def deactivate_media():
    display_message("d","\nActive Media Sources:")
    active_medias = [(section, config[section]['media_name']) for section in config.sections() if config[section]['enable'] == '1']
    
    if not active_medias:
        display_message("w","No active media sources available.")
        return

    table = PrettyTable()
    table.field_names = ["Index", "Media Name", "Media Type", "URL"]
    
    # Set alignment: center index, left-align other columns
    table.align["Index"] = "c"  # Center align index
    table.align["Media Name"] = "l"  # Left align media name
    table.align["Media Type"] = "l"  # Left align media type
    table.align["URL"] = "l"  # Left align URL
    
    for i, (section, media_name) in enumerate(active_medias, 1):
        table.add_row([i, sanitize_input(media_name), config[section]['type'], config[section]['url']])
    
    display_message("d", table)
    
    choice = input("Enter the numbers of the media to deactivate (e.g., '1,2,3') or 'c' to cancel: ").strip()
    if choice.lower() == 'c':
        display_message("d", "Deactivation cancelled.")
        return
    
    try:
        # Convert the user input into a list of indices, handling both single and multiple inputs
        indices = [int(x.strip()) - 1 for x in choice.split(',') if x.strip().isdigit()]
        
        if not indices:
            raise ValueError("No valid selections.")
        
        # Remove duplicates from the input
        indices = list(set(indices))

        # Check for valid index range
        invalid_indices = [i for i in indices if i < 0 or i >= len(active_medias)]
        if invalid_indices:
            raise IndexError(f"Invalid selection(s): {', '.join(map(str, [i + 1 for i in invalid_indices]))}. Please try again.")
        
        # Deactivate the selected media sources
        deactivated_medias = []
        for index in indices:
            section, media_name = active_medias[index]
            config[section]['enable'] = '0'
            deactivated_medias.append(media_name)
        
        # Display success message for deactivated media
        display_message("s", f"The following media sources have been deactivated: {', '.join(deactivated_medias)}.")
        save_config()
    
    except (IndexError, ValueError) as e:
        display_message("e", str(e))
        display_message("e", "Invalid input. Please enter valid numbers separated by commas.")


def show_menu():
    list_summary()
    display_message("d","\nMenu:")
    menu = PrettyTable()
    menu.field_names = ["Index", "Option"]
    
    # Set alignment: center index, left-align option
    menu.align["Index"] = "c"  # Center align index
    menu.align["Option"] = "l"  # Left align option
    
    
    menu.add_row(["1", "Activate Media"])
    menu.add_row(["2", "Deactivate Media"])
    menu.add_row(["3", "List Media"])
    menu.add_row(["4", "Add Media"])
    menu.add_row(["5", "Remove Media"])
    menu.add_row(["0", "Finish"])
    
    display_message("d", menu)

def manage_source():
    while True:
        show_menu()
        choice = input("Select an option: ").strip()
        if choice == '1':
            activate_media()
        elif choice == '2':
            deactivate_media()
        elif choice == '3':
            list_media()
        elif choice == '4':
            add_media()
        elif choice == '5':
            remove_media()
        elif choice == '0':
            break
        else:
            display_message("e", "Invalid option selected.")


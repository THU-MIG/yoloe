import configparser
from prettytable import PrettyTable

config_file = '/apps/deepstream-yolo-e2e/config/python_app/config.ini'

def load_config():
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def get_config():
    config = load_config()
    # Load configuration variables
    muxer_batch_timeout_usec = config.getint('Settings', 'MUXER_BATCH_TIMEOUT_USEC')
    muxer_output_width = config.getint('Settings', 'MUXER_OUTPUT_WIDTH')
    muxer_output_height = config.getint('Settings', 'MUXER_OUTPUT_HEIGHT')

    tiled_output_width = config.getint('Settings', 'TILED_OUTPUT_WIDTH')
    tiled_output_height = config.getint('Settings', 'TILED_OUTPUT_HEIGHT')
    osd_process_mode = config.getint('Settings', 'OSD_PROCESS_MODE')
    osd_display_text = config.getint('Settings', 'OSD_DISPLAY_TEXT')
    rtsp_udpsync = config.getint('Settings', 'RTSP_UDPSYNC')

    rtsp_port = config.getint('Settings', 'RTSP_PORT')  
    rtsp_factory = config.get('Settings', 'RTSP_FACTORY')  
    output_directory = config.get('Settings', 'OUTPUT_DIRECTORY')  
    output_prefix = config.get('Settings', 'OUTPUT_PREFIX')

    return {
        'MUXER_BATCH_TIMEOUT_USEC': muxer_batch_timeout_usec,
        'MUXER_OUTPUT_WIDTH': muxer_output_width,
        'MUXER_OUTPUT_HEIGHT': muxer_output_height,
        'TILED_OUTPUT_WIDTH': tiled_output_width,
        'TILED_OUTPUT_HEIGHT': tiled_output_height,
        'OSD_PROCESS_MODE': osd_process_mode,
        'OSD_DISPLAY_TEXT': osd_display_text,
        'RTSP_UDPSYNC': rtsp_udpsync,
        'RTSP_FACTORY' : rtsp_factory,
        'RTSP_PORT': rtsp_port,
        'OUTPUT_DIRECTORY' : output_directory ,
        'OUTPUT_PREFIX' : output_prefix
    }


def show_current_resolution():
    config = load_config()
    table = PrettyTable()
    table.field_names = ["Output Resolution"]
    resolution = f"{config.getint('Settings', 'tiled_output_width')} x {config.getint('Settings', 'tiled_output_height')}"
    table.add_row([resolution])
    print(table)

def update_resolution(config, width, height):
    config.set('Settings', 'tiled_output_width', str(width))
    config.set('Settings', 'tiled_output_height', str(height))

    with open(config_file, 'w') as configfile:
        config.write(configfile)
    print("Resolution updated to:", width, "x", height)

def show_wide_options():
    resolutions = [
        ("640 x 480 (SD)", 640, 480),
        ("1280 x 720 (HD)", 1280, 720),
        ("1920 x 1080 (Full HD)", 1920, 1080),
        ("2560 x 1440 (QHD)", 2560, 1440),
    ]

    print("\nWide Resolutions:")
    table = PrettyTable()
    table.field_names = ["Option", "Resolution"]
    table.align["Resolution"] = "l"

    for i, res in enumerate(resolutions, start=1):
        table.add_row([i, res[0]])

    print(table)
    return resolutions

def custom_resolution():
    while True:
        print("\nEnter custom resolution (or 'b' to go back):")
        width_input = input("Enter custom width (minimum 640): ")
        if width_input.lower() == 'b':
            return None, None

        height_input = input("Enter custom height (minimum 480): ")
        if height_input.lower() == 'b':
            return None, None

        try:
            width = int(width_input)
            height = int(height_input)

            if width >= 640 and height >= 480:
                return width, height
            else:
                print("Width and height must be at least 640 x 480.")
        except ValueError:
            print("Please enter valid integers for width and height.")

def show_main_menu():
    print("\nCurrent:")
    show_current_resolution()
    print("\nMenu:")
    table = PrettyTable()
    table.field_names = ["Option", "Description"]
    table.align["Description"] = "l"
    table.add_row(["1", "Change resolution"])
    table.add_row(["2", "Finish"])

    print(table)

def show_aspect_ratio_menu():
    print("\nChoose Aspect Ratio:")
    table = PrettyTable()
    table.field_names = ["Option", "Description"]
    table.align["Description"] = "l"
    table.add_row(["1", "Wide"])
    table.add_row(["2", "Custom"])
    
    print(table)

def menu_system_resolution():
    config = load_config()

    while True:
        show_main_menu()  # Chama a função para exibir o menu principal
        choice = input("Choose an option (1-2): ")

        if choice == '1':
            show_aspect_ratio_menu()  # Exibe o menu de proporção
            ratio_choice = input("Select an option (1-2): ")

            if ratio_choice == "1":
                resolutions = show_wide_options()
                resolution_choice = input("Choose a resolution option (or 'b' to go back): ")

                if resolution_choice.lower() == 'b':
                    continue
                
                if resolution_choice.isdigit() and 1 <= int(resolution_choice) <= len(resolutions):
                    width, height = resolutions[int(resolution_choice) - 1][1:3]
                    update_resolution(config, width, height)
                else:
                    print("Invalid choice. Please try again.")

            elif ratio_choice == "2":
                width, height = custom_resolution()
                if width is not None and height is not None:
                    update_resolution(config, width, height)

            else:
                print("Invalid choice. Returning to main menu.")
        elif choice == '2':
            break
        else:
            print("Invalid choice. Please try again.")


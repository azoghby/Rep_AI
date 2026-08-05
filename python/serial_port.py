from serial.tools import list_ports


def find_arduino_port():
    ports = list(list_ports.comports())

    for port in ports:
        if "usbmodem" in port.device.lower():
            print(f"Selected Arduino port: {port.device}")
            return port.device

    print("No Arduino usbmodem port found.")

    if ports:
        print("Available ports:")
        for port in ports:
            print(f"  {port.device} - {port.description}")
    else:
        print("No serial ports are currently available.")

    raise RuntimeError("Could not find an Arduino port containing 'usbmodem'.")

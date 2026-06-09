from ids_peak import ueye

devices = ueye.get_devices()
for dev in devices:
    print(dev)
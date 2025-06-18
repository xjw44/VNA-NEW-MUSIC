# To generate CSV's for control, download the Easywave Software at https://siglentna.com/service-and-support/firmware-software/waveform-generators/
import pyvisa as visa
import tkinter as tk

class SDGController:
    def __init__(self, usb_address='USB0::0xF4EC::0x1102::SDG2XCAC6R0570::INSTR'):
        self.usb_address = usb_address
        self.AWG = self.connect()

    def connect(self):
        rm = visa.ResourceManager()
        device = rm.open_resource(self.usb_address)
        print(device.query('*IDN?'))
        return device

    def reconnect(self):
        self.AWG = self.connect()

    def close(self):
        self.AWG.close()

    def set_output_state(self, channel_number, turn_on):
        try:
            if channel_number not in [1, 2]:
                raise ValueError("Channel number must be 1 or 2")
            state_tuple = ("OFF", "ON")
            state = state_tuple[1] if turn_on else state_tuple[0]
            self.AWG.write(f'C{channel_number}:OUTP {state}')
        except Exception as e:
            print(f"Error setting output state: {e}")
    def get_output_state(self, channel_number):
        try:
            if channel_number not in [1, 2]:
                raise ValueError("Channel number must be 1 or 2")
            return self.AWG.query(f'C{channel_number}:OUTP?')
        except Exception as e:
            print(f"Error getting output state: {e}")
            return None
    def set_frequency_base_wave(self, channel_number, frequency):
        try:
            frequency = int(frequency)
            if frequency > 40e6 or frequency < 0:
                raise ValueError("Frequency cannot exceed 40 MHz and must be 0 or greater")
            if channel_number not in [1, 2]:
                raise ValueError("Channel number must be 1 or 2")
            self.AWG.write(f'C{channel_number}:BSWV FRQ,{frequency}')
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"Error setting frequency base wave: {e}")


    def set_phase_base_wave(self, channel_number, phase):
        try:
            if channel_number not in [1, 2]:
                raise ValueError("Channel number must be 1 or 2")
            phase = int(phase)
            self.AWG.write(f'C{channel_number}:BSWV PHSE,{phase}')
        except Exception as e:
            print(f"Error setting phase base wave: {e}")
    def get_base_wave_state(self, channel_number):
        try:
            if channel_number not in [1, 2]:
                raise ValueError("Channel number must be 1 or 2")
            return self.AWG.query(f'C{channel_number}:BSWV?')
        except Exception as e:
            print(f"Error getting base wave state: {e}")
            return None

    def set_power_base_wave(self, channel_number, power):
        try:
            power = float(power)
            if power > 20 or power < -120:
                raise ValueError("Power must be between -120 dBm and 20 dBm")
            if channel_number not in [1, 2]:
                raise ValueError("Channel number must be 1 or 2")
            power = self.convert_dbm_to_vpp(power)
            self.AWG.write(f'C{channel_number}:BSWV AMP,{power}')
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"Error setting power base wave: {e}")

    def convert_dbm_to_vpp(self, power):
        return 2 * (2 ** 0.5) * (10 ** ((power - 13.0103) / 20))

    def set_dsb_modulation_state(self, channel_number, turn_on):
        try:
            if channel_number not in [1, 2]:
                raise ValueError("Channel number must be 1 or 2")
            state_tuple = ("OFF", "ON")
            state = state_tuple[1] if turn_on else state_tuple[0]
            self.AWG.write(f'C{channel_number}:MDWV STATE, {state}')
        except Exception as e:
            print(f"Error setting DSB modulation state: {e}")

    def set_dsb_modulation_frequency(self, channel_number, frequency):
        try:
            if channel_number not in [1, 2]:
                raise ValueError("Channel number must be 1 or 2")
            frequency = int(frequency)
            if frequency > 40e6 or frequency < 0:
                raise ValueError("Frequency cannot exceed 40 MHz and must be 0 or greater")
            self.AWG.write(f'C{channel_number}:MDWV DSBAM,FRQ,{frequency}')
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"Error setting DSB modulation frequency: {e}")
            
    def get_dsb_modulation_state(self, channel_number):
        try:
            if channel_number not in [1, 2]:
                raise ValueError("Channel number must be 1 or 2")
            return self.AWG.query(f'C{channel_number}:MDWV?')
        except Exception as e:
            print(f"Error getting DSB modulation state: {e}")
            return None
        
    def update_output_state(self, channel, state):
        try:
            self.reconnect()
            self.set_output_state(channel, state == "ON")
            print(self.get_output_state(channel))
        finally:
            self.close()

    def update_frequency_base_wave(self, channel, frequency):
        try:
            self.reconnect()
            self.set_frequency_base_wave(channel, frequency)
            print(self.get_base_wave_state(channel))
        finally:
            self.close()

    def update_dsb_modulation_frequency(self, channel, frequency):
        try:
            self.reconnect()
            self.set_dsb_modulation_frequency(channel, frequency)
            print(self.get_dsb_modulation_state(channel))
        finally:
            self.close()

    def start_up_awg_dsb_modulation(self, f1, power):
        try:
            self.reconnect()
            main_tone = (f1[0] + f1[1]) / 2
            mod_tone = abs((f1[0] - f1[1])) / 2
            print(main_tone, mod_tone)
            self.set_frequency_base_wave(1, main_tone)
            self.set_frequency_base_wave(2, main_tone)
            self.set_dsb_modulation_frequency(1, mod_tone)
            self.set_dsb_modulation_frequency(2, mod_tone)
            self.set_power_base_wave(1, power)
            self.set_power_base_wave(2, power)
            self.set_phase_base_wave(1, 0)
            self.set_phase_base_wave(2, 90)
            self.set_output_state(1, True)
            self.set_output_state(2, True)
        finally:
            self.close()

    def shutdown_awg(self):
        try:
            self.reconnect()
            self.set_output_state(1, False)
            self.set_output_state(2, False)
            print(self.get_output_state(1))
            print(self.get_output_state(2))
        finally:
            self.close()



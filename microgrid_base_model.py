import pandapower as pp
import numpy as np
import json
import os

VN_BASE_KV = 0.12  # 0.12 kV for the 120V system base

# --- NUMERICAL SCALING FACTOR ---
# Scaling is necessary for numerical stability in the Newton-Raphson solver
# due to the extremely small power levels (milliwatts) and low voltages.
POWER_SCALING_FACTOR = 10000.0

# Component ID mapping dictionary
ID_MAP = {"buses": {}, "transformers": {}, "loads": {}, "sensors": {}}

# Total Power in MW (All values are scaled up by POWER_SCALING_FACTOR)
LOAD_LED_STRIPS_MW = (0.0288 / 1000.0) * POWER_SCALING_FACTOR  # 28.8W * 10000
LOAD_AC_MOTORS_MW = (0.0144 / 1000.0) * POWER_SCALING_FACTOR  # 14.4W * 10000
LOAD_LED_DIODES_MW = (0.0024 / 1000.0) * POWER_SCALING_FACTOR  # 2.4W * 10000

# PF for motors (inductive load)
MOTOR_PF = 0.85
MOTOR_S_MVA = LOAD_AC_MOTORS_MW / MOTOR_PF
MOTOR_Q_MVAR = MOTOR_S_MVA * np.sin(np.arccos(MOTOR_PF))


def create_base_network():
    """Initializes network with 16 buses, transformers, and loads."""
    print("Initializing Pandapower Network...")

    # --- FIX 2: Explicitly set a smaller base MVA (e.g., 1000 kVA or 1 MVA) for stability ---
    net = pp.create_empty_network(f_hz=60.0, sn_mva=1.0)

    # --BUS CREATION (16 bus)--

    # Input Grid (120V AC)
    bus_grid = pp.create_bus(net, vn_kv=VN_BASE_KV, name="BUS_GRID_120V")
    ID_MAP["buses"]["BUS_GRID"] = bus_grid

    # Primary Step-Down Output (48V AC & 24V AC)
    bus_t1_48v = pp.create_bus(net, vn_kv=0.048, name="BUS_T1_48V_RES")
    bus_t2_48v = pp.create_bus(net, vn_kv=0.048, name="BUS_T2_48V_COM")
    bus_ss1_24v = pp.create_bus(net, vn_kv=0.024, name="BUS_SS1_24V_IND")
    bus_ss2_24v = pp.create_bus(net, vn_kv=0.024, name="BUS_SS2_24V_HOS")
    ID_MAP["buses"]["BUS_T1_RES"] = bus_t1_48v
    ID_MAP["buses"]["BUS_T2_COM"] = bus_t2_48v
    ID_MAP["buses"]["BUS_SS1_IND"] = bus_ss1_24v
    ID_MAP["buses"]["BUS_SS2_HOS"] = bus_ss2_24v

    # Secondary Transformer Buses (12V AC Rectifier Inputs)
    # total 11 secondary buses (R1-R6, C1-C4, H1) -> 15 buses so far, we split the 24V bus further to create 16 total

    # Residential(R1-R6) - Fed from T1 (48V)
    bus_r = []
    for i in range(1, 7):
        bus_name = f"BUS_R{i}_12V_REC_IN"
        bus_idx = pp.create_bus(net, vn_kv=0.012, name=bus_name)
        bus_r.append(bus_idx)
        ID_MAP["buses"][f"BUS_R{i}"] = bus_idx

    # Commerical/Industrial/Hospital (C1-C5) - Fed from T2/SS1/SS2
    bus_cih = []
    cih_names = ["C1", "C2", "C3", "C4", "H1"]
    for i, name in enumerate(cih_names):
        bus_name = f"BUS_{name}_12V_REC_IN"
        bus_idx = pp.create_bus(net, vn_kv=0.012, name=bus_name)
        bus_cih.append(bus_idx)
        ID_MAP["buses"][f"BUS_{name}"] = bus_idx

    # 16 TOTAL BUSES (1 grid + 4 primary + 6 r-secondary + 5 c/i/h secondary)

    # --EXTERNAL GRID (Source)--
    pp.create_ext_grid(net, bus_grid, vm_pu=1.0, name="EXT_GRID_CPP", s_sc_max_mva=1000.0)

    # --STANDARD TYPES--

    # line type
    # using low impedance values for short lab wires
    line_data = {
        "r_ohm_per_km": 0.5, "x_ohm_per_km": 0.1, "c_nf_per_km": 0.0, "max_i_ka": 0.05, "type": "LV_LAB_LINE"
    }
    pp.create_std_type(net, line_data, name="LV_LAB_LINE", element="line")

    # transformer type
    # vk_percent and vkr_percent represent impedance and resistance respectively
    TRAFO_STD_DATA = {
        "sn_mva": 0.005 * POWER_SCALING_FACTOR,  # 5kVA * 10000 = 50 MVA (for stability)
        "vk_percent": 0.5,  # short circuit voltage (Fixed to 0.5 for stability)
        "vkr_percent": 0.1,  # resistive short circuit (Fixed to 0.1 for stability)
        "pfe_kw": 0.05 * POWER_SCALING_FACTOR,  # iron losses (Scaled)
        "i0_percent": 0.4,  # no load current
        # placeholder voltage fields
        "vn_hv_kv": 0.1,
        "vn_lv_kv": 0.1,
        "shift_degree": 0
    }
    pp.create_std_type(net, TRAFO_STD_DATA, name="DUMMY_TRAFO_STD", element="trafo")

    # --TRANSFORMERS (Primary & Secondary Step-Downs)--

    # Primary Transformers (120V to 48V/24V)
    xfmr_t1 = pp.create_transformer(net, bus_grid, bus_t1_48v, "DUMMY_TRAFO_STD", name="XFMR_T1_120_48V",
                                    vn_hv_kv=VN_BASE_KV, vn_lv_kv=0.048, shift_degree=0)
    xfmr_t2 = pp.create_transformer(net, bus_grid, bus_t2_48v, "DUMMY_TRAFO_STD", name="XFMR_T2_120_48V",
                                    vn_hv_kv=VN_BASE_KV, vn_lv_kv=0.048, shift_degree=0)
    xfmr_ss1 = pp.create_transformer(net, bus_grid, bus_ss1_24v, "DUMMY_TRAFO_STD", name="XFMR_SS1_120_24V",
                                     vn_hv_kv=VN_BASE_KV, vn_lv_kv=0.024, shift_degree=0)
    xfmr_ss2 = pp.create_transformer(net, bus_grid, bus_ss2_24v, "DUMMY_TRAFO_STD", name="XFMR_SS2_120_24V",
                                     vn_hv_kv=VN_BASE_KV, vn_lv_kv=0.024, shift_degree=0)

    ID_MAP["transformers"]["XFMR_T1"] = xfmr_t1
    ID_MAP["transformers"]["XFMR_T2"] = xfmr_t2
    ID_MAP["transformers"]["XFMR_SS1"] = xfmr_ss1
    ID_MAP["transformers"]["XFMR_SS2"] = xfmr_ss2

    # Secondary Transformers (48V/24V to 12V) - 11 total

    # T1(48V) feeds 6 Residential branches (R1-R6)
    for i, bus_lv in enumerate(bus_r):
        xfmr_idx = pp.create_transformer(net, bus_t1_48v, bus_lv, "DUMMY_TRAFO_STD", name=f"XFMR_R{i + 1}_48_12V",
                                         vn_hv_kv=0.048, vn_lv_kv=0.012, shift_degree=0)
        ID_MAP["transformers"][f"XFMR_R{i + 1}"] = xfmr_idx

    # T2/SS1/SS2 feed 5 C/I/H branches (C1-C4, H1)

    # T2(48V) -> Commercial (C1,C2)
    for i in range(2):
        xfmr_idx = pp.create_transformer(net, bus_t2_48v, bus_cih[i], "DUMMY_TRAFO_STD", name=f"XFMR_C{i + 1}_48_12V",
                                         vn_hv_kv=0.048, vn_lv_kv=0.012, shift_degree=0)
        ID_MAP["transformers"][f"XFMR_C{i + 1}"] = xfmr_idx

    # SS1(24V) -> Industrial (C3, C4)
    for i in range(2, 4):
        xfmr_idx = pp.create_transformer(net, bus_ss1_24v, bus_cih[i], "DUMMY_TRAFO_STD", name=f"XFMR_C{i + 1}_24_12V",
                                         vn_hv_kv=0.024, vn_lv_kv=0.012, shift_degree=0)
        ID_MAP["transformers"][f"XFMR_C{i + 1}"] = xfmr_idx

    # SS2(24V) -> Hospital (H1)
    xfmr_idx = pp.create_transformer(net, bus_ss2_24v, bus_cih[4], "DUMMY_TRAFO_STD", name="XFMR_H1_24_12V",
                                     vn_hv_kv=0.024, vn_lv_kv=0.012, shift_degree=0)
    ID_MAP["transformers"]["XFMR_H1"] = xfmr_idx

    # --LOADS--

    # Residential LED Strips (28.8W total, split over 6 buses R1-R6)
    led_strip_per_bus = LOAD_LED_STRIPS_MW / 6.0
    for i, bus_idx in enumerate(bus_r):
        load_idx = pp.create_load(net, bus_idx, p_mw=led_strip_per_bus, q_mvar=0.0, name=f"LOAD_R{i + 1}_LED_STRIP")
        ID_MAP["loads"][f"LOAD_R{i + 1}"] = load_idx

    # LED Diodes (Streetlights, 2.4W total, split over C1-C2, part of commercial/industrial)
    led_diode_per_bus = LOAD_LED_DIODES_MW / 2.0
    for i in range(2):
        load_idx = pp.create_load(net, bus_cih[i], p_mw=led_diode_per_bus, q_mvar=0.0, name=f"LOAD_C{i + 1}_LED_DIODE")
        ID_MAP["loads"][f"LOAD_C{i + 1}_DIODE"] = load_idx

    # AC Motors (14.4W total, connected to 24V substations SS1/SS2)
    motor_per_bus_p = LOAD_AC_MOTORS_MW / 2.0
    motor_per_bus_q = MOTOR_Q_MVAR / 2.0

    # Motor 1 on SS1 Industrial (BUS_SS1_24V_IND)
    load_ss1_idx = pp.create_load(net, bus_ss1_24v, p_mw=motor_per_bus_p, q_mvar=motor_per_bus_q, name="LOAD_SS1_MOTOR")
    ID_MAP["loads"]["LOAD_SS1_MOTOR"] = load_ss1_idx

    # Motor 2 on SS2 Hospital (BUS_SS2_24V_HOS)
    load_ss2_idx = pp.create_load(net, bus_ss2_24v, p_mw=motor_per_bus_p, q_mvar=motor_per_bus_q, name="LOAD_SS2_MOTOR")
    ID_MAP["loads"]["LOAD_SS2_MOTOR"] = load_ss2_idx

    # Other loads (Buck converters, Xmas lights - assumed 5W total, placed on C3)
    other_load_p = (0.005 / 1000.0) * POWER_SCALING_FACTOR  # 5W scaled
    load_c3_idx = pp.create_load(net, bus_cih[2], p_mw=other_load_p, q_mvar=0.0, name="LOAD_C3_MISC")
    ID_MAP["loads"]["LOAD_C3_MISC"] = load_c3_idx

    # --VIRTUAL SENSORS (Measurement Points)--
    ID_MAP["sensors"]["VOLT_SS1_BUS"] = bus_ss1_24v  # ZMPT101B on 24V SS
    ID_MAP["sensors"]["CURRENT_T1_XFMR"] = xfmr_t1  # ACS712 on primary transformer circuit

    print("Network creation complete. Total Buses:", len(net.bus))
    return net


def save_id_map(id_map_data):
    """Saves the component ID map to a JSON file after ensuring all indices are standard Python ints."""

    # Create a clean map to store JSON-compatible data
    clean_id_map = {}

    # Iterate through the top-level keys (buses, transformers, loads, sensors)
    for element_type, id_dict in id_map_data.items():
        cleaned_inner_dict = {}
        # Iterate through names and indices
        for name, index in id_dict.items():
            # Convert NumPy integers (like int64) to standard Python int for JSON serialization
            # This is necessary because Pandapower indices are often np.int64
            cleaned_inner_dict[name] = int(index)
        clean_id_map[element_type] = cleaned_inner_dict

    filepath = "id_mapping.json"
    with open(filepath, 'w') as f:
        json.dump(clean_id_map, f, indent=4)
    print(f"Component ID map saved to {filepath}")


if __name__ == "__main__":
    net = create_base_network()

    # Save the ID map for cross-referencing
    save_id_map(ID_MAP)

    # RUN NORMAL STEADY-STATE OPERATION
    print("\n--- Running Normal Power Flow (Steady State) ---")
    try:
        # Run power flow for normal operating conditions. Increased max_iter to 50.
        pp.runpp(net, max_iter=50)
        print("Power flow completed successfully.")
    except Exception as e:
        print(f"Power flow failed: {e}")
        # Explicitly set converged to False if an exception occurs during the run.
        net.converged = False

    # VALIDATION
    print("\n--- Model Validation ---")

    # Only run validation if the power flow converged successfully
    if net.converged:
        # Merge net.bus (input data) and net.res_bus (results data) to get all columns.
        bus_validation = net.bus.merge(net.res_bus, left_index=True, right_index=True)
        print("Bus Voltages (pu and kV):")
        # Display voltages grouped by nominal voltage
        print(bus_validation.sort_values(by='vn_kv')[["name", "vn_kv", "vm_pu", "vm_kv"]])

        print("\nTransformer Loading (%):")
        # Note: The loading percent is calculated based on the scaled MVA values,
        # but since the loads were scaled equally, the percentage is still correct
        print(net.res_transformer[["name", "loading_percent"]])
    else:
        print("Power flow results are not available to validate the model.")

    pp.to_json(net, "base_microgrid_net.json")
    print("\nBase network structure saved to base_microgrid_net.json")

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import math
import os

def extract_structure_data(atoms):
    """
    ASE の Atoms オブジェクトから必要な構造データを取得し、辞書形式で返す。
    """
    structure_data = {
        "symbols": atoms.get_chemical_symbols(),  # 元素記号
        "positions": atoms.get_positions().tolist(),  # 座標 (Angstrom)
        "cell": atoms.get_cell().tolist()  # 格子ベクトル
    }
    return structure_data

def write_qe_input(input_data, atomic_species, structure_data, filename="qe_input.in", kpts=None, koffset=(0,0,0)):
    """
    Quantum ESPRESSO の入力ファイルを生成する関数。
    `input_data`: 計算パラメータ（辞書形式）
    `atomic_species`: {'元素記号': (原子量, 擬ポテンシャルファイル名)}
    `structure_data`: `extract_structure_data` で取得したデータ(辞書形式)
    `filename`: 出力ファイル名(引数がなければデフォルトで 'qe_input.in' を出力)
    `kpts`: k点グリッド
    `koffset`: k点のオフセット
    """
    sections = ["control", "system", "electrons", "ions", "cell", "fcp", "rism"]

    def format_section(name, data):
        """ 各セクションをフォーマット """
        lines = [f"&{name.upper()}"]
        for key, value in data.items():
            if isinstance(value, bool):  # ✅ Boolean を .TRUE. / .FALSE. に変換
                value = ".TRUE." if value else ".FALSE."
            elif isinstance(value, str) and not (value.startswith(".") and value.endswith(".")):  
                value = f"'{value}'"  # ✅ 文字列は '...' で囲む
            lines.append(f"    {key} = {value}")  # ✅ 変換された値をリストに追加
        lines.append("/")  # ✅ 

        return "\n".join(lines)

    with open(filename, "w") as f:
        for section in sections:
            if section in input_data:
                f.write(format_section(section, input_data[section]) + "\n\n")
        # ATOMIC_SPECIES
        f.write("ATOMIC_SPECIES\n")
        for element, (mass, pseudo) in atomic_species.items():
            f.write(f"{element:<2} {mass:12.6f} {pseudo}\n")
        f.write("\n")
        # ATOMIC_POSITIONS
        f.write("ATOMIC_POSITIONS angstrom\n")
        for symbol, pos in zip(structure_data["symbols"], structure_data["positions"]):
            f.write(f"{symbol:<2} {pos[0]:12.8f} {pos[1]:12.8f} {pos[2]:12.8f}\n")
        f.write("\n")
        # K_POINTS
        if kpts is None:
            f.write("K_POINTS gamma\n")  #  kpts=None の場合
        else:
            f.write("K_POINTS automatic\n")
            koffset = koffset if koffset is not None else (0,0,0)  # ✅ koffsetがNoneならデフォルト (0,0,0) を設定
            f.write(f"{kpts[0]} {kpts[1]} {kpts[2]} {koffset[0]} {koffset[1]} {koffset[2]}\n")
        f.write("\n")
        # CELL_PARAMETERS
        f.write("CELL_PARAMETERS angstrom\n")
        for vec in structure_data["cell"]:
            f.write(f"{vec[0]:12.8f} {vec[1]:12.8f} {vec[2]:12.8f}\n")
        f.write("\n")

    print(f"Quantum ESPRESSO input file '{filename}' has been generated.")
    
def qe_bool(value):
    """ Quantum ESPRESSO 用の Boolean 変換 """
    if isinstance(value, bool):  # まず Boolean型かをチェック
        return ".TRUE." if value else ".FALSE."
    return value  # それ以外の値はそのまま返す    

def parse_qe_xml(prefixes, outdir):
    """
    Quantum ESPRESSO の XML ファイルを解析し、エネルギー・フェルミエネルギー・総電荷・力を取得する関数。

    Parameters:
        prefixes (list): 計算対象の prefix リスト
        outdir (str): XML ファイルが格納されているディレクトリのパス

    Returns:
        dict: 各 prefix ごとのエネルギー、フェルミエネルギー、総電荷、力のデータ
    """
    scf_result = {}
    for prefix in prefixes:
        xml_file = os.path.join(outdir, f"{prefix}.xml")  # XMLファイルのフルパスを作成
        try:
            # XML ファイルを読み込む
            tree = ET.parse(xml_file)
            root = tree.getroot()
            print(f"[SUCCESS] Loaded XML for prefix: {prefix}")
            # 全エネルギーを取得 (Hartree → eV に変換)
            energy_tag = root.find(".//output/total_energy/etot")
            if energy_tag is not None:
                print(f"  [INFO] Parsing total energy data from {xml_file}")
                energy = float(energy_tag.text) * 27.2114
            else:
                energy = None
                print("  [WARNING] `<total_energy/etot>` not found in `<output>`.")
            # Fermi energy を取得 (Hartree → eV に変換)
            fermi_energy_tag = root.find(".//output/band_structure/fermi_energy")
            if fermi_energy_tag is not None:
                print(f"  [INFO] Parsing fermi energy data from {xml_file}")
                fermi_energy = float(fermi_energy_tag.text) * 27.2114
            else:
                fermi_energy = None
                print("  [WARNING] `<fermi_energy>` not found in `<output>`.")
            # 余剰電荷を取得
            tot_charge_tag = root.find(".//input/bands/tot_charge")
            if tot_charge_tag is not None:
                print(f"  [INFO] Parsing tot_charge data from {xml_file}")
                tot_charge = float(tot_charge_tag.text)
            else:
                tot_charge = None
                print("  [WARNING] `<tot_charge>` not found in `<output>`.")
            # ESM
            esm_tag = root.find(".//output/boundary_conditions/esm/bc")
            if esm_tag is not None:
                print(f"  [INFO] Parsing ESM data from {xml_file}")
                esm_bc = esm_tag.text.strip()
            else:
                esm_bc = None
                print("  [WARNING] `<esm>` not found in `<output>`.")
            # 力を取得 (Hartree/Bohr → eV/Å に変換)
            forces_data = []
            forces_section = root.find(".//output/forces")  # `<output>` 内の `<forces>` を取得
            if forces_section is None:
                print(f"  [WARNING] No `<forces>` section found in `<output>` of {xml_file}")
            else:
                forces_text = forces_section.text.strip()
                if not forces_text:
                    print(f"  [WARNING] `<forces>` section is empty in `<output>` of {xml_file}")
                else:
                    print(f"  [INFO] Parsing forces data from {xml_file}")
                    try:
                        force_lines = forces_text.split("\n")
                        for i, line in enumerate(force_lines):
                            force_values = line.strip().split()
                            if len(force_values) == 3:  # 3つの値がある場合のみ処理
                                fx, fy, fz = map(float, force_values)
                                forces_data.append([i+1, fx * 25.711043, fy * 25.711043, fz * 25.711043])
                            else:
                                print(f"  [ERROR] Invalid force line format at index {i+1}: {line}")
                    except ValueError as e:
                        print(f"  [ERROR] Failed to parse force data in {xml_file}: {e}")
            # データを保存 (追加)
            if prefix not in scf_result:
                scf_result[prefix] = {}
            scf_result[prefix]["energy"] = energy         # 全エネルギー (eV)
            scf_result[prefix]["forces"] = forces_data     # 力 (eV/Å)
            scf_result[prefix]["fermi_energy"] = fermi_energy  # Fermi energy (eV)
            scf_result[prefix]["tot_charge"] = tot_charge # 余剰電荷
            scf_result[prefix]["esm_bc"] = esm_bc # ESM
    
        except FileNotFoundError:
            print(f"Warning: {xml_file} not found for prefix {prefix}")
        except Exception as e:
            print(f"Error reading {xml_file}: {e}")
        
    return scf_result
    
def plot_esm1(filename, titlename=''):
    esm1 = np.loadtxt(filename, comments='#')

    fig = plt.figure(
        figsize=(6, 4),  # inch
#       dpi=100,  # dpi
#        edgecolor='black',
#        linewidth='1'
    )

    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.suptitle(titlename)

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.set_xlabel('z (A)')
    ax1.set_ylabel('rho (e/A)')
    ax2.set_xlabel('z (A)')
    ax2.set_ylabel('V_hartree (eV)')
    ax3.set_xlabel('z (A)')
    ax3.set_ylabel('V_ion (eV)')
    ax4.set_xlabel('z (A)')
    ax4.set_ylabel('V_electrostatic (eV)')

    ax4.axhline(0.0, linewidth=1, linestyle='dashed', color='black')

    ax1.plot(esm1[:, 0], esm1[:, 1], color='black', linestyle='solid')
    ax2.plot(esm1[:, 0], esm1[:, 2], color='black', linestyle='solid')
    ax3.plot(esm1[:, 0], esm1[:, 3], color='black', linestyle='solid')
    ax4.plot(esm1[:, 0], esm1[:, 4], color='black', linestyle='solid')

    plt.show()

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
# %load_ext autoreload
# %autoreload 2

# %% tags=["remove-cell"]
import os, shutil, glob, copy, math
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import fcc111, graphene, add_adsorbate, molecule, cut
from ase.constraints import FixAtoms
from ase.geometry import wrap_positions
from ase.io import read, write
from ase.io.cube import read_cube_data
from ase.io.espresso import read_espresso_out
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from ase.units import Bohr
#
root_dir = '/Users/otani/code/'
cmp_host = 'mac'
#cmp_host = 'ohtaka'

# %% [markdown]
# このファイルでは以下のような構成を考えいている。
# - このファイル自体はiCloudで同期されている。
# - ファイルのセーブ時にはoutputセルは消去してファイルサイズは最小を保つ。
# - データはroot_dir下に置かれる。ファイルサイズは大きくなるので、クラウドではなく各マシンのローカルフォルダに置かれていることを想定

# %% [markdown] tags=[]
# # RISM-DEVの計算

# %% [markdown] tags=[]
# ## 本山さんの計算

# %% [markdown] toc-hr-collapsed=true toc-hr-collapsed=true
# ### motoyama/

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# #### Li-EtOH構造の読み込み

# %% [markdown]
# 本山さんの用意した入力ファイルを読み込む。root_dirの下のwork_dirの中でファイルを展開した。

# %%
work_dir = root_dir + 'motoyama/Li-EtOH/'
pprefix = 'Li-EtOH'

# %% [markdown]
# まず、左右が対象の`R8+8`を読み込んで構造データを読み込む

# %%
qe_input = read(work_dir + 'R8+8/' + pprefix + '.in', format='espresso-in')

# %% [markdown]
# 読み込んだ構造を出力する。

# %% tags=[]
aobj = qe_input
print(f' Unit cell: a = ({aobj.cell[0,0]:9.5f}, {aobj.cell[0,1]:9.5f}, {aobj.cell[0,2]:9.5f})')
print(f'            b = ({aobj.cell[1,0]:9.5f}, {aobj.cell[1,1]:9.5f}, {aobj.cell[1,2]:9.5f})')
print(f'            c = ({aobj.cell[2,0]:9.5f}, {aobj.cell[2,1]:9.5f}, {aobj.cell[2,2]:9.5f})')
print(f' Number of atoms: {len(aobj.positions):5d}')
print(f' Species, Positions:')
for i in range(len(aobj.positions)):
    print(f'  \'{aobj.symbols[i]:<2}\' ({aobj.positions[i,0]:9.5f}, {aobj.positions[i,1]:9.5f}, {aobj.positions[i,2]:9.5f})')

# %% [markdown] tags=[] toc-hr-collapsed=true tags=[] toc-hr-collapsed=true
# ### 2022091115/

# %% [markdown] tags=[] toc-hr-collapsed=true tags=[] toc-hr-collapsed=true toc-hr-collapsed=true tags=[] toc-hr-collapsed=true
# #### Li-EtOHのESM計算

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ##### prefix, rood_dir, work_dir設定
# 計算する真空の厚みを8~32Ang.に変化させて計算する。

# %%
prefix = ['r8_bc1', 'r16_bc1', 'r24_bc1', 'r32_bc1']
lvac = 8.0
rvac = [8.0, 16.0, 24.0, 32.0]
work_dir = root_dir + '2022091115/'
if not os.path.exists(work_dir):
    print(work_dir + " not found. Make dir")
    os.makedirs(work_dir)
# QEbranch: rism_expand_oneside 54d66b07
qeroot_dir = '/Users/otani/Program/q-e/'
pseudo_dir = qeroot_dir + 'pseudo/'
temp_dir = qeroot_dir + 'tempdir/'

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### 入力ファイルを生成

# %% tags=[]
for index, item in enumerate(prefix):
    #新しいobjectにqe_inputをコピー
    slab_ESM = copy.deepcopy(qe_input)
    #上の原子位置からスラブの厚みを単純に計算
    slab_thickness = qe_input.positions[2,2] - qe_input.positions[0,2]
    #スラブの左端の原子をz=0にずらす
    slab_ESM.translate((0.0,0.0,qe_input.positions[2,2]))
    #ユニットセルの大きさを再計算
    a = slab_ESM.cell[0,0]
    b = slab_ESM.cell[1,1]
    c = lvac + slab_thickness + rvac[index]
    #ユニットセルの大きさを設定
    slab_ESM.set_cell([a,b,c])
    #ユニットセルの大きさの変更に合わせて、スラブの左端の原子を真空がlvacになるようにする
    slab_ESM.translate((0.0,0.0,lvac))
    #ESM用にずらす
    slab_ESM.translate((0.0,0.0,-slab_ESM.cell[2,2]/2.0))
    #PWの入力を作る
    pseudopotentials = {'Li':'Li.pbe-n-van.UPF'}
    input_data = {
        'control': {
            'calculation' : 'scf',
            'restart_mode': 'from_scratch',
            'pseudo_dir'  : pseudo_dir,
            'outdir'      : temp_dir,
            'prefix'      : prefix[index],
            'tprnfor'     : False,
        },
        'system': {
            'ibrav'          : 0,
            'ntyp'           : 0,
            'nat'            : 5,
            'ecutwfc'        : 50.0,
            'ecutrho'        : 300.0,
            'occupations'    : 'smearing',
            'smearing'       : 'gauss',
            'degauss'        : 0.01,
            'tot_charge'     : 0.0,
            'assume_isolated': 'esm',
            'esm_bc'         : 'bc1',
        },
        'electrons': {
            #'diagonalization': 'rmm-davidson',
            'diagonalization': 'rmm',
            'mixing_beta': 0.4,
        },
    }
    #inputファイルの出力
    inpfile = work_dir + prefix[index] + '.in'
    write(inpfile, slab_ESM, format='espresso-in', input_data=input_data, 
      pseudopotentials=pseudopotentials, kpts=(2, 2, 1), koffset=(0, 0, 0))

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ##### ジョブスクリプトの作成

# %% tags=[]
try:
    cmp_host
except NameError:
    print('please define cmp_host')
else:
    if cmp_host == 'mac':
        for index, item in enumerate(prefix): 
            #計算時の入出力ファイル
            inpfile = prefix[index] + '.in'
            outfile = prefix[index] + '.out'
            #逐次ジョブ投入用シェルスクリプト
            shfile = work_dir + prefix[index] + '.sh'
            with open(shfile, 'w', encoding='UTF-8') as f:
                f.write('#!/usr/bin/env bash\n')
                f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' +
                        ' < ' + inpfile + ' > ' + outfile + ' 2>&1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
                f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
            os.chmod(shfile, 0o744)
        #一気にジョブ投入量シェルスクリプト
        shfile = work_dir + 'all.sh'
        with open(shfile, 'w', encoding='UTF-8') as f:
            f.write('#!/usr/bin/env bash\n')
            for index, item in enumerate(prefix):
                inpfile = prefix[index] + '.in'
                outfile = prefix[index] + '.out'
                f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' + ' < ' +
                        inpfile + ' > ' + outfile + ' 2>&1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
                f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
        os.chmod(shfile, 0o744)
    elif cmp_host == 'ohtaka':
        print('no contents')
    else:
        print('no calculation resorce specified!')

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ##### 構造を確認する。

# %%
aobj = copy.deepcopy(qe_input)
lvac = 8.0
rvac = 32.0
slab_thickness = qe_input.positions[2,2] - qe_input.positions[0,2]
aobj.translate((0.0,0.0,qe_input.positions[2,2]))
a = qe_input.cell[0,0]
b = qe_input.cell[1,1]
c = lvac + slab_thickness + rvac
aobj.set_cell([a,b,c])
aobj.translate((0.0,0.0,lvac))
#aobj.translate((0.0,0.0,-aobj.cell[2,2]/2.0))
print(f' Unit cell: a = ({aobj.cell[0,0]:9.5f}, {aobj.cell[0,1]:9.5f}, {aobj.cell[0,2]:9.5f})')
print(f'            b = ({aobj.cell[1,0]:9.5f}, {aobj.cell[1,1]:9.5f}, {aobj.cell[1,2]:9.5f})')
print(f'            c = ({aobj.cell[2,0]:9.5f}, {aobj.cell[2,1]:9.5f}, {aobj.cell[2,2]:9.5f})')
print(f' Number of atoms: {len(aobj.positions):5d}')
print(f' Species, Positions:')
for i in range(len(aobj.positions)):
    print(f'  \'{aobj.symbols[i]:<2}\' ({aobj.positions[i,0]:9.5f}, {aobj.positions[i,1]:9.5f}, {aobj.positions[i,2]:9.5f})')
fig, ax = plt.subplots(1,1, figsize=(12, 6))
ax.set_axis_off()
#plot_atoms(aobj, ax, radii=0.8, rotation=('0x,0y,0z'))
plot_atoms(aobj, ax, radii=0.8, rotation=('90x,90y,90z'))
#view(slab, viewer='ngl')
#plt.savefig("fig4.eps") 
plt.show()

# %% [markdown] tags=[]
# ##### 結果をプロット

# %% tags=[]
#プロットのデフォルト値にリセット
plt.rcParams.update(plt.rcParamsDefault)

config = {
    "lines.linestyle"     :"solid",
    "lines.linewidth"     :1,
    "xtick.direction"     :"in",
    "xtick.labelbottom"   :True,
    "ytick.labelleft"     :True,
    "xtick.labeltop"      :False,
    "ytick.direction"     :"in",
    "ytick.labelright"    :False,
    "figure.autolayout"   :True, #図の間の空白を自動調整
    #"figure.subplot.hspace: 0.2 #自動調整がうまくいかない場合は直接指定
    #"figure.subplot.wspace: 0.2 #自動調整がうまくいかない場合は直接指定
    "font.size"           :15.0,
    #フォントを選ぶ
    # Helvetica
    #"text.usetex"         : True,
    #"text.latex.preamble" : r'\usepackage[cm]{sfmath}',
    #"font.family"         : 'sans-serif',
    #"font.sans-serif"     : 'Helvetica',
    #})
    # Roman
    #"text.usetex"         : True,
    #"font.family"         : "serif",
    #"font.serif"          : "Times New Roman",
}
#プロットのデフォルト値を変更
plt.rcParams.update(config)

fig = plt.figure()
# figureのタイトル
fig.suptitle("ESM (bc1)")
    
data_esm1 = []
for index, item in enumerate(prefix):
    #読み込むファイルを決める
    filename = work_dir + prefix[index] + '.esm1'
    data = np.loadtxt(filename, comments='#')
    data_esm1.append(data)
# plotの流儀（https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9）
# plotの階層構造（https://qiita.com/ceptree/items/5fb5e9e6f29d214153c9）
# plotの引数（https://own-search-and-study.xyz/2016/08/08/matplotlib-pyplotのplotの全引数を使いこなす/)
# fig.add_subplot(x,y,i), x行、y列のタイルのi番目の図
# axesの追加
ax = fig.add_subplot(2, 1, 1)
# axesのタイトル
#ax.set_title("Electrostatic")
ax.set_xlabel(r"$z~(\mathrm{\AA})$")
ax.set_ylabel(r"$V_\mathrm{electrostatic}~(\mathrm{eV})$")
ax.plot(data_esm1[3][:,0]   ,data_esm1[3][:,4], color='b')
ax.plot(data_esm1[2][:,0]-4 ,data_esm1[2][:,4], color='g')
ax.plot(data_esm1[1][:,0]-8 ,data_esm1[1][:,4], color='r')
ax.plot(data_esm1[0][:,0]-12,data_esm1[0][:,4], color='k')
# https://pystyle.info/matplotlib-legend/
labels = ['R32','R24','R16','R8']
plt.legend(labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=config['font.size']-5, frameon=False)
#y軸の範囲を自動設定にしているが、その時のymaxとyminを取得して、縦線の上限・下限値としたい。
y_min, y_max = ax.get_ylim()
plt.ylim(y_min,y_max)
plt.vlines(-22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 30.86/2- 8, y_min, y_max, linestyle='solid', color='r')
plt.vlines( 38.86/2- 4, y_min, y_max, linestyle='solid', color='g')
plt.vlines( 46.86/2   , y_min, y_max, linestyle='solid', color='b')

ax = fig.add_subplot(2, 1, 2)
# axesのタイトル
#ax.set_title("Electrostatic")
ax.set_xlabel(r"$z~(\mathrm{\AA})$")
ax.set_ylabel(r"$V_\mathrm{electrostatic}~(\mathrm{eV})$")
ax.plot(data_esm1[3][:,0],   data_esm1[3][:,4], color='blue')
ax.plot(data_esm1[2][:,0]-4, data_esm1[2][:,4], color='green')
ax.plot(data_esm1[1][:,0]-8, data_esm1[1][:,4], color='red')
ax.plot(data_esm1[0][:,0]-12,data_esm1[0][:,4], color='black')
plt.ylim(-0.04,0.04)
y_min, y_max = ax.get_ylim()
plt.ylim(y_min,y_max)
plt.vlines(-22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 30.86/2- 8, y_min, y_max, linestyle='solid', color='r')
plt.vlines( 38.86/2- 4, y_min, y_max, linestyle='solid', color='g')
plt.vlines( 46.86/2   , y_min, y_max, linestyle='solid', color='b')
#plt.savefig(work_dir + 'esm.pdf')
plt.show()

# %% [markdown] tags=[] toc-hr-collapsed=true tags=[] jp-MarkdownHeadingCollapsed=true toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true
# #### Li-EtOHの共通メッシュESM-RISM計算

# %% [markdown] tags=[]
# ##### prefix, rood_dir, work_dir設定
# 計算する真空の厚みを8~32Ang.に変化させて計算する。

# %%
prefix = ['r8_rism', 'r16_rism', 'r24_rism', 'r32_rism']
lvac = 8.0
rvac = [8.0, 16.0, 24.0, 32.0]
#rvac = [12.0, 20.0]
work_dir = root_dir + '2022091115/'
if not os.path.exists(work_dir):
    print(work_dir + " not found. Make dir")
    os.makedirs(work_dir)
# QEbranch: rism_expand_oneside 54d66b07
qeroot_dir = '/Users/otani/Program/q-e/'
pseudo_dir = qeroot_dir + 'pseudo/'
temp_dir = qeroot_dir + 'tempdir/'

# %% [markdown] tags=[]
# ##### 入力ファイルを生成

# %% tags=[]
rism_start = [1.0, -3.0, -7.0, -11.0]
for index, item in enumerate(prefix):
    #新しいobjectにqe_inputをコピー
    slab_ESM = copy.deepcopy(qe_input)
    #上の原子位置からスラブの厚みを単純に計算
    slab_thickness = qe_input.positions[2,2] - qe_input.positions[0,2]
    #スラブの左端の原子をz=0にずらす
    slab_ESM.translate((0.0,0.0,qe_input.positions[2,2]))
    #ユニットセルの大きさを再計算
    a = slab_ESM.cell[0,0]
    b = slab_ESM.cell[1,1]
    c = lvac + slab_thickness + rvac[index]
    #ユニットセルの大きさを設定
    slab_ESM.set_cell([a,b,c])
    #ユニットセルの大きさの変更に合わせて、スラブの左端の原子を真空がlvacになるようにする
    slab_ESM.translate((0.0,0.0,lvac))
    #ESM用にずらす
    slab_ESM.translate((0.0,0.0,-slab_ESM.cell[2,2]/2.0))
    #PWの入力を作る
    pseudopotentials = {'Li':'Li.pbe-n-van.UPF'}
    input_data = {
        'control': {
            'calculation' : 'scf',
            'restart_mode': 'from_scratch',
            'pseudo_dir'  : pseudo_dir,
            'outdir'      : temp_dir,
            'prefix'      : prefix[index],
            'tprnfor'     : False,
            'trism'       : True,
        },
        'system': {
            'ibrav'          : 0,
            'ntyp'           : 0,
            'nat'            : 5,
            'ecutwfc'        : 50.0,
            'ecutrho'        : 300.0,
            'occupations'    : 'smearing',
            'smearing'       : 'gauss',
            'degauss'        : 0.01,
            'tot_charge'     : 0.0,
            'assume_isolated': 'esm',
            'esm_bc'         : 'bc1',
        },
        'electrons': {
            #'diagonalization': 'rmm-davidson',
            'diagonalization' : 'rmm',
            'mixing_beta'     : 0.4,
            'startingpot'     : 'atom',
            'startingwfc'     : 'atom+randam',
        },
        'rism': {
            'nsolv'                 : 1,
            'closure'               : 'kh',
            'tempv'                 : 300.0,
            'ecutsolv'              : 144.0,
            'solute_lj(1)'          : 'uff',
            'starting1d'            : 'zero',
            'starting3d'            : 'zero',
            'rism3d_conv_level'     : 0.7,
            'laue_rism_length_unit' : 'angstrom',
            'laue_expand_right'     : 30.0,
            'laue_expand_left'      : 0.0,
            'laue_starting_right'   : rism_start[index],
            'laue_buffer_right'     : 8.0,
        },
    }
    solv_info = {
        'density_unit' : 'mol/L',
        'EtOH' : [-1, 'Ethanol.oplsua.MOL'],
    }
    #1drismの結果があれば、それを読み込むように変更
    if os.path.isfile(temp_dir + prefix[index] + '.save/1d-rism_gvv_r.1.xml'):
        input_data['rism']['starting1d'] = 'file'
    #ESMの結果があれば、それを読み込むように変更
    #if os.path.isfile(temp_dir + esm_prefix[index] + '.save/charge-density.dat'):
    #    input_data['electrons']['startingpot'] = 'file'
    #    if not os.path.exists(temp_dir + prefix[index] + '.save'):
    #        print(temp_dir + prefix[index] + '.save' + " not found. Make dir")
    #        os.makedirs(temp_dir + prefix[index] + '.save')
    #    shutil.copy(temp_dir + esm_prefix[index] + '.save/charge-density.dat', temp_dir + prefix[index] + '.save/')
    #if os.path.isfile(temp_dir + esm_prefix[index] + '.save/wfc1.dat'):
    #    input_data['electrons']['startingwfc'] = 'file'
    #    if not os.path.exists(temp_dir + prefix[index] + '.save'):
    #        print(temp_dir + prefix[index] + '.save' + " not found. Make dir")
    #        os.makedirs(temp_dir + prefix[index] + '.save')
    #    for file in glob.glob(temp_dir + esm_prefix[index] + '.save/wfc*.dat'):
    #        shutil.copy(file, temp_dir + prefix[index] + '.save/')
    #inputファイルの出力
    inpfile = work_dir + prefix[index] + '.in'
    write(inpfile, slab_ESM, format='espresso-in', input_data=input_data, solvents_info=solv_info,
      pseudopotentials=pseudopotentials, kpts=(2, 2, 1), koffset=(0, 0, 0))

# %% [markdown] tags=[]
# ##### ジョブスクリプトの作成

# %% tags=[]
try:
    cmp_host
except NameError:
    print('please define cmp_host')
else:
    if cmp_host == 'mac':
        for index, item in enumerate(prefix): 
            #計算時の入出力ファイル
            inpfile = prefix[index] + '.in'
            outfile = prefix[index] + '.out'
            #逐次ジョブ投入用シェルスクリプト
            shfile = work_dir + prefix[index] + '.sh'
            with open(shfile, 'w', encoding='UTF-8') as f:
                f.write('#!/usr/bin/env bash\n')
                f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' +
                        ' < ' + inpfile + ' > ' + outfile + ' 2>&1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
                f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.1drism ' + work_dir + prefix[index] + '.1drism\n')
                f.write('mv ' + temp_dir + prefix[index] + '.rism1 ' + work_dir + prefix[index] + '.rism1\n')
            os.chmod(shfile, 0o744)
        #一気にジョブ投入量シェルスクリプト
        shfile = work_dir + 'all.sh'
        with open(shfile, 'w', encoding='UTF-8') as f:
            f.write('#!/usr/bin/env bash\n')
            for index, item in enumerate(prefix):
                inpfile = prefix[index] + '.in'
                outfile = prefix[index] + '.out'
                f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' + ' < ' +
                        inpfile + ' > ' + outfile + ' 2>&1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
                f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.1drism ' + work_dir + prefix[index] + '.1drism\n')
                f.write('mv ' + temp_dir + prefix[index] + '.rism1 ' + work_dir + prefix[index] + '.rism1\n')
        os.chmod(shfile, 0o744)
    elif cmp_host == 'ohtaka':
        print('no contents')
    else:
        print('no calculation resorce specified!')

# %% [markdown] tags=[]
# ##### 結果をプロット

# %% tags=[]
#プロットのデフォルト値にリセット
plt.rcParams.update(plt.rcParamsDefault)

config = {
    "lines.linestyle"     :"solid",
    "lines.linewidth"     :1,
    "xtick.direction"     :"in",
    "xtick.labelbottom"   :True,
    "ytick.labelleft"     :True,
    "xtick.labeltop"      :False,
    "ytick.direction"     :"in",
    "ytick.labelright"    :False,
    "figure.autolayout"   :True, #図の間の空白を自動調整
    #"figure.subplot.hspace: 0.2 #自動調整がうまくいかない場合は直接指定
    #"figure.subplot.wspace: 0.2 #自動調整がうまくいかない場合は直接指定
    "font.size"           :15.0,
    #フォントを選ぶ
    # Helvetica
    #"text.usetex"         : True,
    #"text.latex.preamble" : r'\usepackage[cm]{sfmath}',
    #"font.family"         : 'sans-serif',
    #"font.sans-serif"     : 'Helvetica',
    #})
    # Roman
    #"text.usetex"         : True,
    #"font.family"         : "serif",
    #"font.serif"          : "Times New Roman",
}
#プロットのデフォルト値を変更
plt.rcParams.update(config)

fig = plt.figure()
# figureのタイトル
fig.suptitle("ESM")
    
data_esm1 = []
for index, item in enumerate(prefix):
    #読み込むファイルを決める
    filename = work_dir + prefix[index] + '.rism1'
    data = np.loadtxt(filename, comments='#')
    data_esm1.append(data)
# plotの流儀（https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9）
# plotの階層構造（https://qiita.com/ceptree/items/5fb5e9e6f29d214153c9）
# plotの引数（https://own-search-and-study.xyz/2016/08/08/matplotlib-pyplotのplotの全引数を使いこなす/)
# fig.add_subplot(x,y,i), x行、y列のタイルのi番目の図
# axesの追加
ax = fig.add_subplot(2, 1, 1)
# axesのタイトル
#ax.set_title("Electrostatic")
ax.set_xlabel(r"$z~(\mathrm{\AA})$")
ax.set_ylabel(r"$V_\mathrm{electrostatic}~(\mathrm{eV})$")
ax.plot(data_esm1[3][:,0]   ,data_esm1[3][:,4], color='b')
ax.plot(data_esm1[2][:,0]-4 ,data_esm1[2][:,4], color='g')
ax.plot(data_esm1[1][:,0]-8 ,data_esm1[1][:,4], color='r')
ax.plot(data_esm1[0][:,0]-12,data_esm1[0][:,4], color='k')
# https://pystyle.info/matplotlib-legend/
labels = ['R32','R24','R16','R8']
plt.legend(labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=config['font.size']-5, frameon=False)
#y軸の範囲を自動設定にしているが、その時のymaxとyminを取得して、縦線の上限・下限値としたい。
y_min, y_max = ax.get_ylim()
plt.ylim(y_min,y_max)
plt.vlines(-22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 30.86/2- 8, y_min, y_max, linestyle='solid', color='r')
plt.vlines( 38.86/2- 4, y_min, y_max, linestyle='solid', color='g')
plt.vlines( 46.86/2   , y_min, y_max, linestyle='solid', color='b')
plt.vlines( 22.86/2-12+30.0, y_min, y_max, linestyle='dotted', color='k')
plt.vlines( 30.86/2- 8+30.0, y_min, y_max, linestyle='dotted', color='r')
plt.vlines( 38.86/2- 4+30.0, y_min, y_max, linestyle='dotted', color='g')
plt.vlines( 46.86/2   +30.0  , y_min, y_max, linestyle='dotted', color='b')

ax = fig.add_subplot(2, 1, 2)
# axesのタイトル
#ax.set_title("Electrostatic")
ax.set_xlabel(r"$z~(\mathrm{\AA})$")
ax.set_ylabel(r"$V_\mathrm{electrostatic}~(\mathrm{eV})$")
ax.plot(data_esm1[3][:,0],   data_esm1[3][:,4], color='blue')
ax.plot(data_esm1[2][:,0]-4, data_esm1[2][:,4], color='green')
ax.plot(data_esm1[1][:,0]-8, data_esm1[1][:,4], color='red')
ax.plot(data_esm1[0][:,0]-12,data_esm1[0][:,4], color='black')
plt.ylim(-0.04,0.04)
y_min, y_max = ax.get_ylim()
plt.ylim(y_min,y_max)
plt.vlines(-22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 30.86/2- 8, y_min, y_max, linestyle='solid', color='r')
plt.vlines( 38.86/2- 4, y_min, y_max, linestyle='solid', color='g')
plt.vlines( 46.86/2   , y_min, y_max, linestyle='solid', color='b')
plt.vlines( 22.86/2-12+30.0, y_min, y_max, linestyle='dotted', color='k')
plt.vlines( 30.86/2- 8+30.0, y_min, y_max, linestyle='dotted', color='r')
plt.vlines( 38.86/2- 4+30.0, y_min, y_max, linestyle='dotted', color='g')
plt.vlines( 46.86/2   +30.0  , y_min, y_max, linestyle='dotted', color='b')
#plt.savefig(work_dir + 'esm.pdf')
plt.show()

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] toc-hr-collapsed=true toc-hr-collapsed=true toc-hr-collapsed=true
# #### Li-EtOHの個別メッシュESM-RISM計算

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### prefix, rood_dir, work_dir設定
# rvac=8.0のセルでESM計算で収束した電子状態を使って本山さんの開発したコードでRISM領域を広げながら計算する。

# %%
prefix = ['r8+8_rism', 'r8+16_rism', 'r8+24_rism']
lvac = 8.0
rvac = 8.0
work_dir = root_dir + '2022091115/'
if not os.path.exists(work_dir):
    print(work_dir + " not found. Make dir")
    os.makedirs(work_dir)
# QEbranch: rism_expand_oneside 54d66b07
qeroot_dir = '/Users/otani/Program/q-e/'
pseudo_dir = qeroot_dir + 'pseudo/'
temp_dir = qeroot_dir + 'tempdir/'

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### 入力ファイルを生成

# %% tags=[]
rism_length = [8.0, 16.0, 24.0]
for index, item in enumerate(prefix):
    #新しいobjectにqe_inputをコピー
    slab_ESM = copy.deepcopy(qe_input)
    #上の原子位置からスラブの厚みを単純に計算
    slab_thickness = qe_input.positions[2,2] - qe_input.positions[0,2]
    #スラブの左端の原子をz=0にずらす
    slab_ESM.translate((0.0,0.0,qe_input.positions[2,2]))
    #ユニットセルの大きさを再計算
    a = slab_ESM.cell[0,0]
    b = slab_ESM.cell[1,1]
    c = lvac + slab_thickness + rvac
    #ユニットセルの大きさを設定
    slab_ESM.set_cell([a,b,c])
    #ユニットセルの大きさの変更に合わせて、スラブの左端の原子を真空がlvacになるようにする
    slab_ESM.translate((0.0,0.0,lvac))
    #ESM用にずらす
    slab_ESM.translate((0.0,0.0,-slab_ESM.cell[2,2]/2.0))
    #PWの入力を作る
    pseudopotentials = {'Li':'Li.pbe-n-van.UPF'}
    input_data = {
        'control': {
            'calculation' : 'scf',
            'restart_mode': 'from_scratch',
            'pseudo_dir'  : pseudo_dir,
            'outdir'      : temp_dir,
            'prefix'      : prefix[index],
            'tprnfor'     : False,
            'trism'       : True,
        },
        'system': {
            'ibrav'          : 0,
            'ntyp'           : 0,
            'nat'            : 5,
            'ecutwfc'        : 50.0,
            'ecutrho'        : 300.0,
            'occupations'    : 'smearing',
            'smearing'       : 'gauss',
            'degauss'        : 0.01,
            'tot_charge'     : 0.0,
            'assume_isolated': 'esm',
            'esm_bc'         : 'bc1',
        },
        'electrons': {
            #'diagonalization': 'rmm-davidson',
            'diagonalization' : 'rmm',
            'mixing_beta'     : 0.4,
            'startingpot'     : 'atom',
            'startingwfc'     : 'atom+randam',
        },
        'rism': {
            'nsolv'                  : 1,
            'closure'                : 'kh',
            'tempv'                  : 300.0,
            'ecutsolv'               : 144.0,
            'solute_lj(1)'           : 'uff',
            'starting1d'             : 'zero',
            'starting3d'             : 'zero',
            'rism3d_conv_level'      : 0.7,
            'laue_rism_length_unit'  : 'angstrom',
            'laue_expand_right'      : 30.0,
            'laue_expand_left'       : 0.0,
            'laue_starting_right'    : 1.0,
            'laue_buffer_right'      : 8.0,
            'laue_rism_length_right' : rism_length[index],
        },
    }
    solv_info = {
        'density_unit' : 'mol/L',
        'EtOH' : [-1, 'Ethanol.oplsua.MOL'],
    }
    #1drismの結果があれば、それを読み込むように変更
    if os.path.isfile(temp_dir + prefix[index] + '.save/1d-rism_gvv_r.1.xml'):
        input_data['rism']['starting1d'] = 'file'
    #ESMの結果があれば、それを読み込むように変更
    #if os.path.isfile(temp_dir + esm_prefix + '.save/charge-density.dat'):
    #    input_data['electrons']['startingpot'] = 'file'
    #    if not os.path.exists(temp_dir + prefix[index] + '.save'):
    #        print(temp_dir + prefix[index] + '.save' + " not found. Make dir")
    #        os.makedirs(temp_dir + prefix[index] + '.save')
    #    shutil.copy(temp_dir + esm_prefix + '.save/charge-density.dat', temp_dir + prefix[index] + '.save/')
    #if os.path.isfile(temp_dir + esm_prefix + '.save/wfc1.dat'):
    #    input_data['electrons']['startingwfc'] = 'file'
    #    if not os.path.exists(temp_dir + prefix[index] + '.save'):
    #        print(temp_dir + prefix[index] + '.save' + " not found. Make dir")
    #        os.makedirs(temp_dir + prefix[index] + '.save')
    #    for file in glob.glob(temp_dir + esm_prefix + '.save/wfc*.dat'):
    #        shutil.copy(file, temp_dir + prefix[index] + '.save/')
    #inputファイルの出力
    inpfile = work_dir + prefix[index] + '.in'
    write(inpfile, slab_ESM, format='espresso-in', input_data=input_data, solvents_info=solv_info,
      pseudopotentials=pseudopotentials, kpts=(2, 2, 1), koffset=(0, 0, 0))

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### ジョブスクリプトの作成

# %% tags=[]
try:
    cmp_host
except NameError:
    print('please define cmp_host')
else:
    if cmp_host == 'mac':
        for index, item in enumerate(prefix): 
            #計算時の入出力ファイル
            inpfile = prefix[index] + '.in'
            outfile = prefix[index] + '.out'
            #逐次ジョブ投入用シェルスクリプト
            shfile = work_dir + prefix[index] + '.sh'
            with open(shfile, 'w', encoding='UTF-8') as f:
                f.write('#!/usr/bin/env bash\n')
                f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' +
                        ' < ' + inpfile + ' > ' + outfile + ' 2>&1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
                f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.1drism ' + work_dir + prefix[index] + '.1drism\n')
                f.write('mv ' + temp_dir + prefix[index] + '.rism1 ' + work_dir + prefix[index] + '.rism1\n')
            os.chmod(shfile, 0o744)
        #一気にジョブ投入量シェルスクリプト
        shfile = work_dir + 'all.sh'
        with open(shfile, 'w', encoding='UTF-8') as f:
            f.write('#!/usr/bin/env bash\n')
            for index, item in enumerate(prefix):
                inpfile = prefix[index] + '.in'
                outfile = prefix[index] + '.out'
                f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' + ' < ' +
                        inpfile + ' > ' + outfile + ' 2>&1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
                f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.1drism ' + work_dir + prefix[index] + '.1drism\n')
                f.write('mv ' + temp_dir + prefix[index] + '.rism1 ' + work_dir + prefix[index] + '.rism1\n')
        os.chmod(shfile, 0o744)
    elif cmp_host == 'ohtaka':
        print('no contents')
    else:
        print('no calculation resorce specified!')

# %% [markdown]
# ##### 結果をプロット

# %% tags=[]
#プロットのデフォルト値にリセット
plt.rcParams.update(plt.rcParamsDefault)

config = {
    "lines.linestyle"     :"solid",
    "lines.linewidth"     :1,
    "xtick.direction"     :"in",
    "xtick.labelbottom"   :True,
    "ytick.labelleft"     :True,
    "xtick.labeltop"      :False,
    "ytick.direction"     :"in",
    "ytick.labelright"    :False,
    "figure.autolayout"   :True, #図の間の空白を自動調整
    #"figure.subplot.hspace: 0.2 #自動調整がうまくいかない場合は直接指定
    #"figure.subplot.wspace: 0.2 #自動調整がうまくいかない場合は直接指定
    "font.size"           :15.0,
    #フォントを選ぶ
    # Helvetica
    #"text.usetex"         : True,
    #"text.latex.preamble" : r'\usepackage[cm]{sfmath}',
    #"font.family"         : 'sans-serif',
    #"font.sans-serif"     : 'Helvetica',
    #})
    # Roman
    #"text.usetex"         : True,
    #"font.family"         : "serif",
    #"font.serif"          : "Times New Roman",
}
#プロットのデフォルト値を変更
plt.rcParams.update(config)

fig = plt.figure()
# figureのタイトル
fig.suptitle("ESM")
    
data_esm1 = []
for index, item in enumerate(prefix):
    #読み込むファイルを決める
    filename = work_dir + prefix[index] + '.rism1'
    data = np.loadtxt(filename, comments='#')
    data_esm1.append(data)
# R8_rism.rism1を読み込んで、R8+0_rism.rism1としてプロット
filename = work_dir + 'R8_rism.rism1'
data = np.loadtxt(filename, comments='#')
data_esm1.insert(0, data)
# plotの流儀（https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9）
# plotの階層構造（https://qiita.com/ceptree/items/5fb5e9e6f29d214153c9）
# plotの引数（https://own-search-and-study.xyz/2016/08/08/matplotlib-pyplotのplotの全引数を使いこなす/)
# fig.add_subplot(x,y,i), x行、y列のタイルのi番目の図
# axesの追加
ax = fig.add_subplot(2, 1, 1)
# axesのタイトル
#ax.set_title("Electrostatic")
ax.set_xlabel(r"$z~(\mathrm{\AA})$")
ax.set_ylabel(r"$V_\mathrm{electrostatic}~(\mathrm{eV})$")
ax.plot(data_esm1[3][:,0]-12,data_esm1[3][:,4], color='b')
ax.plot(data_esm1[2][:,0]-12 ,data_esm1[2][:,4], color='g')
ax.plot(data_esm1[1][:,0]-12 ,data_esm1[1][:,4], color='r')
ax.plot(data_esm1[0][:,0]-12,data_esm1[0][:,4], color='k')
# https://pystyle.info/matplotlib-legend/
labels = ['R8','R16','R24','R32']
plt.legend(labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=config['font.size']-5, frameon=False)
#y軸の範囲を自動設定にしているが、その時のymaxとyminを取得して、縦線の上限・下限値としたい。
y_min, y_max = ax.get_ylim()
plt.ylim(y_min,y_max)
plt.vlines(-22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 30.86/2- 8, y_min, y_max, linestyle='solid', color='r')
plt.vlines( 38.86/2- 4, y_min, y_max, linestyle='solid', color='g')
plt.vlines( 46.86/2   , y_min, y_max, linestyle='solid', color='b')
plt.vlines( 22.86/2-12+30.0, y_min, y_max, linestyle='dotted', color='k')
plt.vlines( 30.86/2- 8+30.0, y_min, y_max, linestyle='dotted', color='r')
plt.vlines( 38.86/2- 4+30.0, y_min, y_max, linestyle='dotted', color='g')
plt.vlines( 46.86/2   +30.0  , y_min, y_max, linestyle='dotted', color='b')

ax = fig.add_subplot(2, 1, 2)
# axesのタイトル
#ax.set_title("Electrostatic")
ax.set_xlabel(r"$z~(\mathrm{\AA})$")
ax.set_ylabel(r"$V_\mathrm{electrostatic}~(\mathrm{eV})$")
ax.plot(data_esm1[3][:,0]-12,   data_esm1[3][:,4], color='blue')
ax.plot(data_esm1[2][:,0]-12, data_esm1[2][:,4], color='green')
ax.plot(data_esm1[1][:,0]-12, data_esm1[1][:,4], color='red')
ax.plot(data_esm1[0][:,0]-12,data_esm1[0][:,4], color='black')
plt.ylim(-0.04,0.04)
y_min, y_max = ax.get_ylim()
plt.ylim(y_min,y_max)
plt.vlines(-22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 30.86/2- 8, y_min, y_max, linestyle='solid', color='r')
plt.vlines( 38.86/2- 4, y_min, y_max, linestyle='solid', color='g')
plt.vlines( 46.86/2   , y_min, y_max, linestyle='solid', color='b')
plt.vlines( 22.86/2-12+30.0, y_min, y_max, linestyle='dotted', color='k')
plt.vlines( 30.86/2- 8+30.0, y_min, y_max, linestyle='dotted', color='r')
plt.vlines( 38.86/2- 4+30.0, y_min, y_max, linestyle='dotted', color='g')
plt.vlines( 46.86/2   +30.0  , y_min, y_max, linestyle='dotted', color='b')
#plt.savefig(work_dir + 'esm.pdf')
plt.show()

# %% [markdown] toc-hr-collapsed=true tags=[] toc-hr-collapsed=true toc-hr-collapsed=true
# ### 2022091715/

# %% [markdown] tags=[] toc-hr-collapsed=true tags=[] toc-hr-collapsed=true jp-MarkdownHeadingCollapsed=true toc-hr-collapsed=true
# #### Li-EtOHのESM計算(lrism-6.1を使う)

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### prefix, rood_dir, work_dir設定
# 計算する真空の厚みを8~32Ang.に変化させて計算する。

# %%
prefix = ['r8_bc1', 'r16_bc1', 'r24_bc1', 'r32_bc1']
lvac = 8.0
rvac = [8.0, 16.0, 24.0, 32.0]
work_dir = root_dir + '2022091715/'
if not os.path.exists(work_dir):
    print(work_dir + " not found. Make dir")
    os.makedirs(work_dir)
# QEbranch: rism_expand_oneside 54d66b07
qeroot_dir = '/Users/otani/Program/q-e/worktree/lrism-6.1/'
pseudo_dir = qeroot_dir + 'pseudo/'
temp_dir = qeroot_dir + 'tempdir/'

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### 入力ファイルを生成

# %% tags=[]
for index, item in enumerate(prefix):
    #新しいobjectにqe_inputをコピー
    slab_ESM = copy.deepcopy(qe_input)
    #上の原子位置からスラブの厚みを単純に計算
    slab_thickness = qe_input.positions[2,2] - qe_input.positions[0,2]
    #スラブの左端の原子をz=0にずらす
    slab_ESM.translate((0.0,0.0,qe_input.positions[2,2]))
    #ユニットセルの大きさを再計算
    a = slab_ESM.cell[0,0]
    b = slab_ESM.cell[1,1]
    c = lvac + slab_thickness + rvac[index]
    #ユニットセルの大きさを設定
    slab_ESM.set_cell([a,b,c])
    #ユニットセルの大きさの変更に合わせて、スラブの左端の原子を真空がlvacになるようにする
    slab_ESM.translate((0.0,0.0,lvac))
    #ESM用にずらす
    slab_ESM.translate((0.0,0.0,-slab_ESM.cell[2,2]/2.0))
    #PWの入力を作る
    pseudopotentials = {'Li':'Li.pbe-n-van.UPF'}
    input_data = {
        'control': {
            'calculation' : 'scf',
            'restart_mode': 'from_scratch',
            'pseudo_dir'  : pseudo_dir,
            'outdir'      : temp_dir,
            'prefix'      : prefix[index],
            'tprnfor'     : False,
        },
        'system': {
            'ibrav'          : 0,
            'ntyp'           : 0,
            'nat'            : 5,
            'ecutwfc'        : 50.0,
            'ecutrho'        : 300.0,
            'occupations'    : 'smearing',
            'smearing'       : 'gauss',
            'degauss'        : 0.01,
            'tot_charge'     : 0.0,
            'assume_isolated': 'esm',
            'esm_bc'         : 'bc1',
        },
        'electrons': {
            #'diagonalization': 'rmm-davidson',
            'diagonalization': 'rmm',
            'mixing_beta': 0.4,
        },
    }
    #inputファイルの出力
    inpfile = work_dir + prefix[index] + '.in'
    write(inpfile, slab_ESM, format='espresso-in', input_data=input_data, 
      pseudopotentials=pseudopotentials, kpts=(2, 2, 1), koffset=(0, 0, 0))

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### ジョブスクリプトの作成

# %% tags=[]
try:
    cmp_host
except NameError:
    print('please define cmp_host')
else:
    if cmp_host == 'mac':
        for index, item in enumerate(prefix): 
            #計算時の入出力ファイル
            inpfile = prefix[index] + '.in'
            outfile = prefix[index] + '.out'
            #逐次ジョブ投入用シェルスクリプト
            shfile = work_dir + prefix[index] + '.sh'
            with open(shfile, 'w', encoding='UTF-8') as f:
                f.write('#!/usr/bin/env bash\n')
                f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' +
                        ' < ' + inpfile + ' > ' + outfile + ' 2>&1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
            os.chmod(shfile, 0o744)
        #一気にジョブ投入量シェルスクリプト
        shfile = work_dir + 'all.sh'
        with open(shfile, 'w', encoding='UTF-8') as f:
            f.write('#!/usr/bin/env bash\n')
            for index, item in enumerate(prefix):
                inpfile = prefix[index] + '.in'
                outfile = prefix[index] + '.out'
                f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' + ' < ' +
                        inpfile + ' > ' + outfile + ' 2>&1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
        os.chmod(shfile, 0o744)
    elif cmp_host == 'ohtaka':
        print('no contents')
    else:
        print('no calculation resorce specified!')

# %% [markdown] tags=[]
# ##### 結果をプロット

# %% tags=[]
#プロットのデフォルト値にリセット
plt.rcParams.update(plt.rcParamsDefault)

config = {
    "lines.linestyle"     :"solid",
    "lines.linewidth"     :1,
    "xtick.direction"     :"in",
    "xtick.labelbottom"   :True,
    "ytick.labelleft"     :True,
    "xtick.labeltop"      :False,
    "ytick.direction"     :"in",
    "ytick.labelright"    :False,
    "figure.autolayout"   :True, #図の間の空白を自動調整
    #"figure.subplot.hspace: 0.2 #自動調整がうまくいかない場合は直接指定
    #"figure.subplot.wspace: 0.2 #自動調整がうまくいかない場合は直接指定
    "font.size"           :15.0,
    #フォントを選ぶ
    # Helvetica
    #"text.usetex"         : True,
    #"text.latex.preamble" : r'\usepackage[cm]{sfmath}',
    #"font.family"         : 'sans-serif',
    #"font.sans-serif"     : 'Helvetica',
    #})
    # Roman
    #"text.usetex"         : True,
    #"font.family"         : "serif",
    #"font.serif"          : "Times New Roman",
}
#プロットのデフォルト値を変更
plt.rcParams.update(config)

fig = plt.figure()
# figureのタイトル
fig.suptitle("ESM (bc1)")
    
data_esm1 = []
for index, item in enumerate(prefix):
    #読み込むファイルを決める
    filename = work_dir + prefix[index] + '.esm1'
    data = np.loadtxt(filename, comments='#')
    data_esm1.append(data)
# plotの流儀（https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9）
# plotの階層構造（https://qiita.com/ceptree/items/5fb5e9e6f29d214153c9）
# plotの引数（https://own-search-and-study.xyz/2016/08/08/matplotlib-pyplotのplotの全引数を使いこなす/)
# fig.add_subplot(x,y,i), x行、y列のタイルのi番目の図
# axesの追加
ax = fig.add_subplot(2, 1, 1)
# axesのタイトル
#ax.set_title("Electrostatic")
ax.set_xlabel(r"$z~(\mathrm{\AA})$")
ax.set_ylabel(r"$V_\mathrm{electrostatic}~(\mathrm{eV})$")
ax.plot(data_esm1[3][:,0]   ,data_esm1[3][:,4], color='b')
ax.plot(data_esm1[2][:,0]-4 ,data_esm1[2][:,4], color='g')
ax.plot(data_esm1[1][:,0]-8 ,data_esm1[1][:,4], color='r')
ax.plot(data_esm1[0][:,0]-12,data_esm1[0][:,4], color='k')
# https://pystyle.info/matplotlib-legend/
labels = ['R32','R24','R16','R8']
plt.legend(labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=config['font.size']-5, frameon=False)
#y軸の範囲を自動設定にしているが、その時のymaxとyminを取得して、縦線の上限・下限値としたい。
y_min, y_max = ax.get_ylim()
plt.ylim(y_min,y_max)
plt.vlines(-22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 30.86/2- 8, y_min, y_max, linestyle='solid', color='r')
plt.vlines( 38.86/2- 4, y_min, y_max, linestyle='solid', color='g')
plt.vlines( 46.86/2   , y_min, y_max, linestyle='solid', color='b')

ax = fig.add_subplot(2, 1, 2)
# axesのタイトル
#ax.set_title("Electrostatic")
ax.set_xlabel(r"$z~(\mathrm{\AA})$")
ax.set_ylabel(r"$V_\mathrm{electrostatic}~(\mathrm{eV})$")
ax.plot(data_esm1[3][:,0],   data_esm1[3][:,4], color='blue')
ax.plot(data_esm1[2][:,0]-4, data_esm1[2][:,4], color='green')
ax.plot(data_esm1[1][:,0]-8, data_esm1[1][:,4], color='red')
ax.plot(data_esm1[0][:,0]-12,data_esm1[0][:,4], color='black')
plt.ylim(-0.1,0.1)
y_min, y_max = ax.get_ylim()
plt.ylim(y_min,y_max)
plt.vlines(-22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 30.86/2- 8, y_min, y_max, linestyle='solid', color='r')
plt.vlines( 38.86/2- 4, y_min, y_max, linestyle='solid', color='g')
plt.vlines( 46.86/2   , y_min, y_max, linestyle='solid', color='b')
#plt.savefig(work_dir + 'esm.pdf')
plt.show()

# %% [markdown] tags=[] toc-hr-collapsed=true tags=[] toc-hr-collapsed=true
# #### Li-EtOHの共通メッシュESM-RISM計算(lrism-6.1を使う)

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### prefix, rood_dir, work_dir設定
# 計算する真空の厚みを8~32Ang.に変化させて計算する。

# %%
prefix = ['r8_rism', 'r16_rism', 'r24_rism', 'r32_rism']
lvac = 8.0
rvac = [8.0, 16.0, 24.0, 32.0]
#rvac = [12.0, 20.0]
work_dir = root_dir + '2022091715/'
if not os.path.exists(work_dir):
    print(work_dir + " not found. Make dir")
    os.makedirs(work_dir)
# QEbranch: rism_expand_oneside 54d66b07
qeroot_dir = '/Users/otani/Program/q-e/worktree/lrism-6.1/'
pseudo_dir = qeroot_dir + 'pseudo/'
temp_dir = qeroot_dir + 'tempdir/'

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### 入力ファイルを生成

# %% tags=[]
#rism_start = [1.0, -3.0, -7.0, -11.0]
rism_start = [-0.1, -3.0, -7.0, -11.0]
for index, item in enumerate(prefix):
    #新しいobjectにqe_inputをコピー
    slab_ESM = copy.deepcopy(qe_input)
    #上の原子位置からスラブの厚みを単純に計算
    slab_thickness = qe_input.positions[2,2] - qe_input.positions[0,2]
    #スラブの左端の原子をz=0にずらす
    slab_ESM.translate((0.0,0.0,qe_input.positions[2,2]))
    #ユニットセルの大きさを再計算
    a = slab_ESM.cell[0,0]
    b = slab_ESM.cell[1,1]
    c = lvac + slab_thickness + rvac[index]
    #ユニットセルの大きさを設定
    slab_ESM.set_cell([a,b,c])
    #ユニットセルの大きさの変更に合わせて、スラブの左端の原子を真空がlvacになるようにする
    slab_ESM.translate((0.0,0.0,lvac))
    #ESM用にずらす
    slab_ESM.translate((0.0,0.0,-slab_ESM.cell[2,2]/2.0))
    #PWの入力を作る
    pseudopotentials = {'Li':'Li.pbe-n-van.UPF'}
    input_data = {
        'control': {
            'calculation' : 'scf',
            'restart_mode': 'from_scratch',
            'pseudo_dir'  : pseudo_dir,
            'outdir'      : temp_dir,
            'prefix'      : prefix[index],
            'tprnfor'     : False,
            'trism'       : True,
        },
        'system': {
            'ibrav'          : 0,
            'ntyp'           : 0,
            'nat'            : 5,
            'ecutwfc'        : 50.0,
            'ecutrho'        : 300.0,
            'occupations'    : 'smearing',
            'smearing'       : 'gauss',
            'degauss'        : 0.01,
            'tot_charge'     : 0.0,
            'assume_isolated': 'esm',
            'esm_bc'         : 'bc1',
        },
        'electrons': {
            #'diagonalization': 'rmm-davidson',
            'diagonalization' : 'rmm',
            'mixing_beta'     : 0.4,
            'startingpot'     : 'atom',
            'startingwfc'     : 'atom+randam',
        },
        'rism': {
            'nsolv'                 : 1,
            'closure'               : 'kh',
            'tempv'                 : 300.0,
            'ecutsolv'              : 144.0,
            'solute_lj(1)'          : 'uff',
            'starting1d'            : 'zero',
            'starting3d'            : 'zero',
            'laue_bc'               : 'VSS',
            'rism3d_conv_level'     : 0.7,
            'laue_starting_right'   : rism_start[index]/Bohr, #ang2bohr
            'laue_buffer_right'     : 8.0/Bohr, #ang2bohr
        },
    }
    solv_info = {
        'density_unit' : 'mol/L',
        'EtOH' : [-1, 'Ethanol.oplsua.MOL'],
    }
    #1drismの結果があれば、それを読み込むように変更
    if os.path.isfile(temp_dir + prefix[index] + '.save/1d-rism_gvv_r.1.xml'):
        input_data['rism']['starting1d'] = 'file'
    #ESMの結果があれば、それを読み込むように変更
    #if os.path.isfile(temp_dir + esm_prefix[index] + '.save/charge-density.dat'):
    #    input_data['electrons']['startingpot'] = 'file'
    #    if not os.path.exists(temp_dir + prefix[index] + '.save'):
    #        print(temp_dir + prefix[index] + '.save' + " not found. Make dir")
    #        os.makedirs(temp_dir + prefix[index] + '.save')
    #    shutil.copy(temp_dir + esm_prefix[index] + '.save/charge-density.dat', temp_dir + prefix[index] + '.save/')
    #if os.path.isfile(temp_dir + esm_prefix[index] + '.save/wfc1.dat'):
    #    input_data['electrons']['startingwfc'] = 'file'
    #    if not os.path.exists(temp_dir + prefix[index] + '.save'):
    #        print(temp_dir + prefix[index] + '.save' + " not found. Make dir")
    #        os.makedirs(temp_dir + prefix[index] + '.save')
    #    for file in glob.glob(temp_dir + esm_prefix[index] + '.save/wfc*.dat'):
    #        shutil.copy(file, temp_dir + prefix[index] + '.save/')
    #inputファイルの出力
    inpfile = work_dir + prefix[index] + '.in'
    write(inpfile, slab_ESM, format='espresso-in', input_data=input_data, solvents_info=solv_info,
      pseudopotentials=pseudopotentials, kpts=(2, 2, 1), koffset=(0, 0, 0))

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### ジョブスクリプトの作成

# %% tags=[]
try:
    cmp_host
except NameError:
    print('please define cmp_host')
else:
    if cmp_host == 'mac':
        for index, item in enumerate(prefix): 
            #計算時の入出力ファイル
            inpfile = prefix[index] + '.in'
            outfile = prefix[index] + '.out'
            #逐次ジョブ投入用シェルスクリプト
            shfile = work_dir + prefix[index] + '.sh'
            with open(shfile, 'w', encoding='UTF-8') as f:
                f.write('#!/usr/bin/env bash\n')
                f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' +
                        ' < ' + inpfile + ' > ' + outfile + ' 2>&1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.1drism ' + work_dir + prefix[index] + '.1drism\n')
                f.write('mv ' + temp_dir + prefix[index] + '.rism1 ' + work_dir + prefix[index] + '.rism1\n')
            os.chmod(shfile, 0o744)
        #一気にジョブ投入量シェルスクリプト
        shfile = work_dir + 'all.sh'
        with open(shfile, 'w', encoding='UTF-8') as f:
            f.write('#!/usr/bin/env bash\n')
            for index, item in enumerate(prefix):
                inpfile = prefix[index] + '.in'
                outfile = prefix[index] + '.out'
                f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' + ' < ' +
                        inpfile + ' > ' + outfile + ' 2>&1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.1drism ' + work_dir + prefix[index] + '.1drism\n')
                f.write('mv ' + temp_dir + prefix[index] + '.rism1 ' + work_dir + prefix[index] + '.rism1\n')
        os.chmod(shfile, 0o744)
    elif cmp_host == 'ohtaka':
        print('no contents')
    else:
        print('no calculation resorce specified!')

# %% [markdown] tags=[]
# ##### 結果をプロット

# %% tags=[]
#プロットのデフォルト値にリセット
prefix = ['r8_rism', 'r16_rism', 'r24_rism', 'r32_rism']
plt.rcParams.update(plt.rcParamsDefault)

config = {
    "lines.linestyle"     :"solid",
    "lines.linewidth"     :1,
    "xtick.direction"     :"in",
    "xtick.labelbottom"   :True,
    "ytick.labelleft"     :True,
    "xtick.labeltop"      :False,
    "ytick.direction"     :"in",
    "ytick.labelright"    :False,
    "figure.autolayout"   :True, #図の間の空白を自動調整
    #"figure.subplot.hspace: 0.2 #自動調整がうまくいかない場合は直接指定
    #"figure.subplot.wspace: 0.2 #自動調整がうまくいかない場合は直接指定
    "font.size"           :15.0,
    #フォントを選ぶ
    # Helvetica
    #"text.usetex"         : True,
    #"text.latex.preamble" : r'\usepackage[cm]{sfmath}',
    #"font.family"         : 'sans-serif',
    #"font.sans-serif"     : 'Helvetica',
    #})
    # Roman
    #"text.usetex"         : True,
    #"font.family"         : "serif",
    #"font.serif"          : "Times New Roman",
}
#プロットのデフォルト値を変更
plt.rcParams.update(config)

fig = plt.figure()
# figureのタイトル
fig.suptitle("RISM (no-expand)")
    
data_rism1 = []
for index, item in enumerate(prefix):
    #読み込むファイルを決める
    filename = work_dir + prefix[index] + '.rism1'
    data = np.loadtxt(filename, comments='#')
    data_rism1.append(data)
# plotの流儀（https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9）
# plotの階層構造（https://qiita.com/ceptree/items/5fb5e9e6f29d214153c9）
# plotの引数（https://own-search-and-study.xyz/2016/08/08/matplotlib-pyplotのplotの全引数を使いこなす/)
# fig.add_subplot(x,y,i), x行、y列のタイルのi番目の図
# axesの追加
ax = fig.add_subplot(2, 1, 1)
# axesのタイトル
#ax.set_title("Electrostatic")
ax.set_xlabel(r"$z~(\mathrm{\AA})$")
ax.set_ylabel(r"$V_\mathrm{electrostatic}~(\mathrm{eV})$")
ax.plot(data_rism1[3][:,0]   ,data_rism1[3][:,4], color='b')
ax.plot(data_rism1[2][:,0]-4 ,data_rism1[2][:,4], color='g')
ax.plot(data_rism1[1][:,0]-8 ,data_rism1[1][:,4], color='r')
ax.plot(data_rism1[0][:,0]-12,data_rism1[0][:,4], color='k')
# https://pystyle.info/matplotlib-legend/
labels = ['R32','R24','R16','R8']
plt.legend(labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=config['font.size']-5, frameon=False)
#y軸の範囲を自動設定にしているが、その時のymaxとyminを取得して、縦線の上限・下限値としたい。
y_min, y_max = ax.get_ylim()
plt.ylim(y_min,y_max)
plt.vlines(-22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 30.86/2- 8, y_min, y_max, linestyle='solid', color='r')
plt.vlines( 38.86/2- 4, y_min, y_max, linestyle='solid', color='g')
plt.vlines( 46.86/2   , y_min, y_max, linestyle='solid', color='b')

ax = fig.add_subplot(2, 1, 2)
# axesのタイトル
#ax.set_title("Electrostatic")
ax.set_xlabel(r"$z~(\mathrm{\AA})$")
ax.set_ylabel(r"$V_\mathrm{electrostatic}~(\mathrm{eV})$")
ax.plot(data_rism1[3][:,0],   data_rism1[3][:,4], color='blue')
ax.plot(data_rism1[2][:,0]-4, data_rism1[2][:,4], color='green')
ax.plot(data_rism1[1][:,0]-8, data_rism1[1][:,4], color='red')
ax.plot(data_rism1[0][:,0]-12,data_rism1[0][:,4], color='black')
plt.ylim(-0.04,0.04)
y_min, y_max = ax.get_ylim()
plt.ylim(y_min,y_max)
plt.vlines(-22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 22.86/2-12, y_min, y_max, linestyle='solid', color='k')
plt.vlines( 30.86/2- 8, y_min, y_max, linestyle='solid', color='r')
plt.vlines( 38.86/2- 4, y_min, y_max, linestyle='solid', color='g')
plt.vlines( 46.86/2   , y_min, y_max, linestyle='solid', color='b')
#plt.savefig(work_dir + 'esm.pdf')
plt.show()

# %% [markdown] toc-hr-collapsed=true
# ### Janakの定理確認(ESM)

# %%
prefix = ['m03', 'm02', 'm01', 'pm0', 'p01', 'p02', 'p03']
tot_chg = [-0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03]
lvac = 8.0
rvac = 8.0
work_dir = root_dir + '2022091811/'
if not os.path.exists(work_dir):
    print(work_dir + " not found. Make dir")
    os.makedirs(work_dir)
# QEbranch: rism_expand_oneside 54d66b07
qeroot_dir = '/Users/otani/Program/q-e/'
pseudo_dir = qeroot_dir + 'pseudo/'
temp_dir = qeroot_dir + 'tempdir/'

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### 入力ファイルを生成

# %% tags=[]
for index, item in enumerate(prefix):
    #新しいobjectにqe_inputをコピー
    slab_ESM = copy.deepcopy(qe_input)
    #上の原子位置からスラブの厚みを単純に計算
    slab_thickness = qe_input.positions[2,2] - qe_input.positions[0,2]
    #スラブの左端の原子をz=0にずらす
    slab_ESM.translate((0.0,0.0,qe_input.positions[2,2]))
    #ユニットセルの大きさを再計算
    a = slab_ESM.cell[0,0]
    b = slab_ESM.cell[1,1]
    c = lvac + slab_thickness + rvac
    #ユニットセルの大きさを設定
    slab_ESM.set_cell([a,b,c])
    #ユニットセルの大きさの変更に合わせて、スラブの左端の原子を真空がlvacになるようにする
    slab_ESM.translate((0.0,0.0,lvac))
    #ESM用にずらす
    slab_ESM.translate((0.0,0.0,-slab_ESM.cell[2,2]/2.0))
    #PWの入力を作る
    pseudopotentials = {'Li':'Li.pbe-n-van.UPF'}
    input_data = {
        'control': {
            'calculation' : 'scf',
            'restart_mode': 'from_scratch',
            'pseudo_dir'  : pseudo_dir,
            'outdir'      : temp_dir,
            'prefix'      : prefix[index],
            'tprnfor'     : False,
        },
        'system': {
            'ibrav'          : 0,
            'ntyp'           : 0,
            'nat'            : 5,
            'ecutwfc'        : 50.0,
            'ecutrho'        : 300.0,
            'occupations'    : 'smearing',
            'smearing'       : 'gauss',
            'degauss'        : 0.01,
            'tot_charge'     : tot_chg[index],
            'assume_isolated': 'esm',
            'esm_bc'         : 'bc3',
        },
        'electrons': {
            #'diagonalization': 'rmm-davidson',
            'diagonalization': 'rmm',
            'mixing_beta': 0.4,
        },
    }
    #inputファイルの出力
    inpfile = work_dir + prefix[index] + '.in'
    write(inpfile, slab_ESM, format='espresso-in', input_data=input_data, 
      pseudopotentials=pseudopotentials, kpts=(24, 24, 1), koffset=(1, 1, 0))

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ##### ジョブスクリプトの作成

# %% tags=[]
try:
    cmp_host
except NameError:
    print('please define cmp_host')
else:
    if cmp_host == 'mac':
        for index, item in enumerate(prefix): 
            #計算時の入出力ファイル
            inpfile = prefix[index] + '.in'
            outfile = prefix[index] + '.out'
            #逐次ジョブ投入用シェルスクリプト
            shfile = work_dir + prefix[index] + '.sh'
            with open(shfile, 'w', encoding='UTF-8') as f:
                f.write('#!/usr/bin/env bash\n')
                f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 2' +
                        ' < ' + inpfile + ' > ' + outfile + ' 2>&1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
                f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
            os.chmod(shfile, 0o744)
        #一気にジョブ投入量シェルスクリプト
        shfile = work_dir + 'all.sh'
        with open(shfile, 'w', encoding='UTF-8') as f:
            f.write('#!/usr/bin/env bash\n')
            for index, item in enumerate(prefix):
                inpfile = prefix[index] + '.in'
                outfile = prefix[index] + '.out'
                f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 2' + ' < ' +
                        inpfile + ' > ' + outfile + ' 2>&1\n')
                f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
                f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
        os.chmod(shfile, 0o744)
    elif cmp_host == 'ohtaka':
        print('no contents')
    else:
        print('no calculation resorce specified!')

# %% [markdown] tags=[]
# ##### 結果をプロット

# %% tags=[]
#プロットのデフォルト値にリセット
plt.rcParams.update(plt.rcParamsDefault)

config = {
    "lines.linestyle"     :"solid",
    "lines.linewidth"     :1,
    "xtick.direction"     :"in",
    "xtick.labelbottom"   :True,
    "ytick.labelleft"     :True,
    "xtick.labeltop"      :False,
    "ytick.direction"     :"in",
    "ytick.labelright"    :False,
    "figure.autolayout"   :True, #図の間の空白を自動調整
    #"figure.subplot.hspace: 0.2 #自動調整がうまくいかない場合は直接指定
    #"figure.subplot.wspace: 0.2 #自動調整がうまくいかない場合は直接指定
    "font.size"           :15.0,
    #フォントを選ぶ
    # Helvetica
    #"text.usetex"         : True,
    #"text.latex.preamble" : r'\usepackage[cm]{sfmath}',
    #"font.family"         : 'sans-serif',
    #"font.sans-serif"     : 'Helvetica',
    #})
    # Roman
    #"text.usetex"         : True,
    #"font.family"         : "serif",
    #"font.serif"          : "Times New Roman",
}
#プロットのデフォルト値を変更
plt.rcParams.update(config)

fig = plt.figure()
# figureのタイトル
fig.suptitle("ESM (bc3)")
    
data_esm1 = []
for index, item in enumerate(prefix):
    #読み込むファイルを決める
    filename = work_dir + prefix[index] + '.esm1'
    data = np.loadtxt(filename, comments='#')
    data_esm1.append(data)
# plotの流儀（https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9）
# plotの階層構造（https://qiita.com/ceptree/items/5fb5e9e6f29d214153c9）
# plotの引数（https://own-search-and-study.xyz/2016/08/08/matplotlib-pyplotのplotの全引数を使いこなす/)
# fig.add_subplot(x,y,i), x行、y列のタイルのi番目の図
# axesの追加
ax = fig.add_subplot(2, 1, 1)
# axesのタイトル
#ax.set_title("Electrostatic")
ax.set_xlabel(r"$z~(\mathrm{\AA})$")
ax.set_ylabel(r"$V_\mathrm{electrostatic}~(\mathrm{eV})$")
ax.plot(data_esm1[6][:,0],data_esm1[6][:,4])
ax.plot(data_esm1[5][:,0],data_esm1[5][:,4])
ax.plot(data_esm1[4][:,0],data_esm1[4][:,4])
ax.plot(data_esm1[3][:,0],data_esm1[3][:,4])
ax.plot(data_esm1[2][:,0],data_esm1[2][:,4])
ax.plot(data_esm1[1][:,0],data_esm1[1][:,4])
ax.plot(data_esm1[0][:,0],data_esm1[0][:,4])
# https://pystyle.info/matplotlib-legend/
labels = ['m03', 'm02', 'm01', 'pm0', 'p01', 'p02', 'p03']
plt.legend(labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=config['font.size']-5, frameon=False)
#データを読み込む
data_fermi = np.loadtxt(work_dir + 'chg-fermi.dat')
# axesの追加
ax = fig.add_subplot(2, 1, 2)
ax.set_xlabel(r"$\Delta N$")
ax.set_ylabel('Fermi energy (eV)')
ax.plot(data_fermi[:,0], data_fermi[:,1], marker='o')
#plt.savefig(work_dir + 'esm.pdf')
plt.show()

# %%

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### 図を描く(plot_1drism, plot_esm1, plot_rism1を利用）

# %%
plot_1drism(work_dir + prefix[0] + '.1drism', '1drism file')


# %%
plot_esm1(work_dir + prefix[0] + '.esm1','esm1 file')

# %%
plot_rism1(work_dir + prefix[0] + '.rism1','rism1 file')

# %% tags=[]
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math

#update font in figure
plt.rcParams.update(plt.rcParamsDefault)
# Helvetica
#plt.rcParams.update({
#    "text.usetex": True,
#    "text.latex.preamble": r'\usepackage[cm]{sfmath}',
#    "font.family": 'sans-serif',
#    "font.sans-serif": 'Helvetica',
#})
# Roman
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    "font.serif": "Times New Roman",
#})

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

    ax1.set_xlabel(r"$z~(\mathrm{\AA})$")
    ax1.set_ylabel(r"$\rho~(e/\mathrm{\AA})$")
    ax2.set_xlabel(r"$z~(\mathrm{\AA})$")
    ax2.set_ylabel(r"$V_\mathrm{hartree}~(\mathrm{eV})$")
    ax3.set_xlabel(r"$z~(\mathrm{\AA})$")
    ax3.set_ylabel(r"$V_\mathrm{ion}~(\mathrm{eV})$")
    ax4.set_xlabel(r"$z~(\mathrm{\AA})$")
    ax4.set_ylabel(r"$V_\mathrm{electrostatic}~(\mathrm{eV})$")

    ax4.axhline(0.0, linewidth=1, linestyle='dashed', color='black')

    ax1.plot(esm1[:, 0], esm1[:, 1], color='black', linestyle='solid')
    ax2.plot(esm1[:, 0], esm1[:, 2], color='black', linestyle='solid')
    ax3.plot(esm1[:, 0], esm1[:, 3], color='black', linestyle='solid')
    ax4.plot(esm1[:, 0], esm1[:, 4], color='black', linestyle='solid')

    #    plt.savefig('%s.pdf'%titlename)
    plt.show()
    
def plot_rism1(filename, titlename=''):
    with open(filename, 'r') as file:
        data_rism1 = file.readlines()
        line_data = data_rism1[1].split()

    data_rism1 = np.loadtxt(filename, comments='#', skiprows=2)

    line_data = line_data[1:]
    new_line_data = [line_data[0] + ' ' + line_data[1]]
    for i in range(2, len(line_data), 3):
        new_line_data.append(line_data[i] + ' ' + line_data[i + 1] + ' ' + line_data[i + 2])

    number_of_sublines = int(math.ceil((len(new_line_data) - 1) / 3))
    number_of_subplots = len(new_line_data) - 1

    fig = plt.figure(
        figsize=(15, 2 * number_of_sublines),  # inch
#        dpi=100,  # dpi
#       edgecolor='black',
#        linewidth=1
    )

    fig.subplots_adjust(wspace=0.5, hspace=0.75)
    fig.suptitle(titlename)

    for n_plot in range(1, number_of_subplots + 1):
        ax1 = fig.add_subplot(number_of_sublines, 3, n_plot)
        ax1.set_xlabel(new_line_data[0])
        ax1.set_ylabel(new_line_data[n_plot])
        # ax1.set_title(line_atoms[n_plot])
        ax1.plot(data_rism1[:, 0], data_rism1[:, n_plot], color='black', linestyle='solid')

    #    plt.savefig('%s.pdf'%titlename)
    plt.show()

def plot_1drism(filename, titlename='', normalization=False, max_x=None):
    with open(filename, 'r') as file:
        data_1drism = file.readlines()
        # line_molecules = data_1drism[3].split()
        line_atoms = data_1drism[4].split()

    data_1drism = np.loadtxt(filename, comments='#', skiprows=5)

    number_of_sublines = int(math.ceil((len(line_atoms) - 1) / 3))
    number_of_subplots = len(line_atoms) - 1

    fig = plt.figure(
        figsize=(15, 2 * number_of_sublines),  # inch
#        dpi=100,  # dpi
#        edgecolor='black',
#        linewidth=1
    )

    fig.subplots_adjust(wspace=0.5, hspace=0.75)
    fig.suptitle(titlename)

    factor_norm = 1

    for n_plot in range(1, number_of_subplots + 1):
        if normalization:
            factor_norm = data_1drism[-1, n_plot]

        ax1 = fig.add_subplot(number_of_sublines, 3, n_plot)
        if max_x is not None:
            ax1.set_xlim([0, max_x])

        ax1.set_xlabel('r (A)')
        ax1.set_ylabel('rdf')

        if (number_of_subplots % 3) == 0:
            if n_plot <= (number_of_subplots // 3 - 1) * 3:
                ax1.set_xlabel('')
            # ax1.set_xticklabels([])

        ax1.set_title(line_atoms[n_plot])
        ax1.plot(data_1drism[:, 0], data_1drism[:, n_plot] / factor_norm, color='black', linestyle='solid')

    #    plt.savefig('%s.pdf'%titlename)
    plt.show()


# %% [markdown]
# ESM計算で収束した電子状態を使ってRISMのみを計算する。

# %% tags=[]
prefix = ['r8_rism_nscf', 'r16_rism_nscf', 'r24_rism_nscf', 'r32_rism_nscf', 'r40_rism_nscf']
lvac = 8.0
rvac = [8.0, 16.0, 24.0, 32.0, 40.0]
work_dir = root_dir + '2022091115/'
if not os.path.exists(work_dir):
    print(work_dir + " not found. Make dir")
    os.makedirs(work_dir)
# QEbranch: rism_expand_oneside 54d66b07
qeroot_dir = '/Users/otani/Program/q-e/'
pseudo_dir = qeroot_dir + 'pseudo/'
temp_dir = qeroot_dir + 'tempdir/'

# %% tags=[]
rism_start = [1.0, -3.0, -7.0, -11.0, -15.0]
for index, item in enumerate(prefix):
    #新しいobjectにqe_inputをコピー
    slab_ESM = copy.deepcopy(qe_input)
    #上の原子位置からスラブの厚みを単純に計算
    slab_thickness = qe_input.positions[2,2] - qe_input.positions[0,2]
    #スラブの左端の原子をz=0にずらす
    slab_ESM.translate((0.0,0.0,qe_input.positions[2,2]))
    #ユニットセルの大きさを再計算
    a = slab_ESM.cell[0,0]
    b = slab_ESM.cell[1,1]
    c = lvac + slab_thickness + rvac[index]
    #ユニットセルの大きさを設定
    slab_ESM.set_cell([a,b,c])
    #ユニットセルの大きさの変更に合わせて、スラブの左端の原子を真空がlvacになるようにする
    slab_ESM.translate((0.0,0.0,lvac))
    #ESM用にずらす
    slab_ESM.translate((0.0,0.0,-slab_ESM.cell[2,2]/2.0))
    #PWの入力を作る
    pseudopotentials = {'Li':'Li.pbe-n-van.UPF'}
    input_data = {
        'control': {
            'calculation' : 'scf',
            'restart_mode': 'from_scratch',
            'pseudo_dir'  : pseudo_dir,
            'outdir'      : temp_dir,
            'prefix'      : prefix[index],
            'tprnfor'     : False,
            'trism'       : True,
        },
        'system': {
            'ibrav'          : 0,
            'ntyp'           : 0,
            'nat'            : 5,
            'ecutwfc'        : 50.0,
            'ecutrho'        : 300.0,
            'occupations'    : 'smearing',
            'smearing'       : 'gauss',
            'degauss'        : 0.01,
            'tot_charge'     : 0.0,
            'assume_isolated': 'esm',
            'esm_bc'         : 'bc1',
        },
        'electrons': {
            #'diagonalization': 'rmm-davidson',
            'diagonalization' : 'rmm',
            'mixing_beta'     : 0.4,
            'conv_thr'        : 1000.0,
            'startingpot'     : 'file',
            'startingwfc'     : 'file',
        },
        'rism': {
            'nsolv'                 : 1,
            'closure'               : 'kh',
            'tempv'                 : 300.0,
            'ecutsolv'              : 144.0,
            'solute_lj(1)'          : 'uff',
            'starting1d'            : 'zero',
            'starting3d'            : 'zero',
            'laue_rism_length_unit' : 'angstrom',
            'laue_expand_right'     : 30.0,
            'laue_expand_left'      : 0.0,
            'laue_starting_right'   : rism_start[index],
            'laue_buffer_right'     : 8.0,
            'rism3d_conv_level'     : 1.0,
        },
    }
    solv_info = {
        'density_unit' : 'mol/L',
        'EtOH' : [-1, 'Ethanol.oplsua.MOL'],
    }
    #1drismの結果があれば、それを読み込むように変更
    if os.path.isfile(temp_dir + prefix[index] + '.save/1d-rism_gvv_r.1.xml'):
        input_data['rism']['starting1d'] = 'file'
    #inputファイルの出力
    inpfile = work_dir + prefix[index] + '.in'
    write(inpfile, slab_ESM, format='espresso-in', input_data=input_data, solvents_info=solv_info,
      pseudopotentials=pseudopotentials, kpts=(2, 2, 1), koffset=(0, 0, 0))
    #計算時の入出力ファイル
    inpfile = prefix[index] + '.in'
    outfile = prefix[index] + '.out'
    #逐次ジョブ投入用シェルスクリプト
    shfile = work_dir + prefix[index] + '.sh'
    with open(shfile, 'w', encoding='UTF-8') as f:
        f.write('#!/usr/bin/env bash\n')
        f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' +
                ' < ' + inpfile + ' > ' + outfile + ' 2>&1\n')
        f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
        f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
        f.write('mv ' + temp_dir + prefix[index] + '.1drism ' + work_dir + prefix[index] + '.1drism\n')
        f.write('mv ' + temp_dir + prefix[index] + '.rism1 ' + work_dir + prefix[index] + '.rism1\n')
    os.chmod(shfile, 0o744)
#一気にジョブ投入量シェルスクリプト
shfile = work_dir + 'all.sh'
with open(shfile, 'w', encoding='UTF-8') as f:
    f.write('#!/usr/bin/env bash\n')
    for index, item in enumerate(prefix):
        inpfile = prefix[index] + '.in'
        outfile = prefix[index] + '.out'
        f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' + ' < ' +
                inpfile + ' > ' + outfile + ' 2>&1\n')
        f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
        f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
        f.write('mv ' + temp_dir + prefix[index] + '.1drism ' + work_dir + prefix[index] + '.1drism\n')
        f.write('mv ' + temp_dir + prefix[index] + '.rism1 ' + work_dir + prefix[index] + '.rism1\n')
os.chmod(shfile, 0o744)

# %% [markdown]
# ESM計算で収束した電子状態を使って本山さんの開発したコードでRISMのみを計算する。

# %%
prefix = ['r8+8_nscf', 'r8+16_nscf', 'r8+24_nscf', 'r8+32_nscf', 'r8+40_nscf']
lvac = 8.0
rvac = 8.0
work_dir = root_dir + '2022091115/'
if not os.path.exists(work_dir):
    print(work_dir + " not found. Make dir")
    os.makedirs(work_dir)
# QEbranch: rism_expand_oneside 54d66b07
qeroot_dir = '/Users/otani/Program/q-e/'
pseudo_dir = qeroot_dir + 'pseudo/'
temp_dir = qeroot_dir + 'tempdir/'

# %% tags=[]
rism_length = [8.0, 16.0, 24.0, 32.0, 40.0]
for index, item in enumerate(prefix):
    #新しいobjectにqe_inputをコピー
    slab_ESM = copy.deepcopy(qe_input)
    #上の原子位置からスラブの厚みを単純に計算
    slab_thickness = qe_input.positions[2,2] - qe_input.positions[0,2]
    #スラブの左端の原子をz=0にずらす
    slab_ESM.translate((0.0,0.0,qe_input.positions[2,2]))
    #ユニットセルの大きさを再計算
    a = slab_ESM.cell[0,0]
    b = slab_ESM.cell[1,1]
    c = lvac + slab_thickness + rvac
    #ユニットセルの大きさを設定
    slab_ESM.set_cell([a,b,c])
    #ユニットセルの大きさの変更に合わせて、スラブの左端の原子を真空がlvacになるようにする
    slab_ESM.translate((0.0,0.0,lvac))
    #ESM用にずらす
    slab_ESM.translate((0.0,0.0,-slab_ESM.cell[2,2]/2.0))
    #PWの入力を作る
    pseudopotentials = {'Li':'Li.pbe-n-van.UPF'}
    input_data = {
        'control': {
            'calculation' : 'scf',
            'restart_mode': 'from_scratch',
            'pseudo_dir'  : pseudo_dir,
            'outdir'      : temp_dir,
            'prefix'      : prefix[index],
            'tprnfor'     : False,
            'trism'       : True,
        },
        'system': {
            'ibrav'          : 0,
            'ntyp'           : 0,
            'nat'            : 5,
            'ecutwfc'        : 50.0,
            'ecutrho'        : 300.0,
            'occupations'    : 'smearing',
            'smearing'       : 'gauss',
            'degauss'        : 0.01,
            'tot_charge'     : 0.0,
            'assume_isolated': 'esm',
            'esm_bc'         : 'bc1',
        },
        'electrons': {
            #'diagonalization': 'rmm-davidson',
            'diagonalization' : 'rmm',
            'mixing_beta'     : 0.4,
            'conv_thr'        : 1000.0,
            'startingpot'     : 'file',
            'startingwfc'     : 'file',
        },
        'rism': {
            'nsolv'                  : 1,
            'closure'                : 'kh',
            'tempv'                  : 300.0,
            'ecutsolv'               : 144.0,
            'solute_lj(1)'           : 'uff',
            'starting1d'             : 'zero',
            'starting3d'             : 'zero',
            'laue_rism_length_unit'  : 'angstrom',
            'laue_expand_right'      : 30.0,
            'laue_expand_left'       : 0.0,
            'laue_starting_right'    : 1.0,
            'laue_buffer_right'      : 8.0,
            'rism3d_conv_level'      : 1.0,
            'laue_rism_length_right' : rism_length[index],
        },
    }
    solv_info = {
        'density_unit' : 'mol/L',
        'EtOH' : [-1, 'Ethanol.oplsua.MOL'],
    }
    #1drismの結果があれば、それを読み込むように変更
    if os.path.isfile(temp_dir + prefix[index] + '.save/1d-rism_gvv_r.1.xml'):
        input_data['rism']['starting1d'] = 'file'
    #inputファイルの出力
    inpfile = work_dir + prefix[index] + '.in'
    write(inpfile, slab_ESM, format='espresso-in', input_data=input_data, solvents_info=solv_info,
      pseudopotentials=pseudopotentials, kpts=(2, 2, 1), koffset=(0, 0, 0))
    #計算時の入出力ファイル
    inpfile = prefix[index] + '.in'
    outfile = prefix[index] + '.out'
    #逐次ジョブ投入用シェルスクリプト
    shfile = work_dir + prefix[index] + '.sh'
    with open(shfile, 'w', encoding='UTF-8') as f:
        f.write('#!/usr/bin/env bash\n')
        f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' +
                ' < ' + inpfile + ' > ' + outfile + ' 2>&1\n')
        f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
        f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
        f.write('mv ' + temp_dir + prefix[index] + '.1drism ' + work_dir + prefix[index] + '.1drism\n')
        f.write('mv ' + temp_dir + prefix[index] + '.rism1 ' + work_dir + prefix[index] + '.rism1\n')
    os.chmod(shfile, 0o744)
#一気にジョブ投入量シェルスクリプト
shfile = work_dir + 'all.sh'
with open(shfile, 'w', encoding='UTF-8') as f:
    f.write('#!/usr/bin/env bash\n')
    for index, item in enumerate(prefix):
        inpfile = prefix[index] + '.in'
        outfile = prefix[index] + '.out'
        f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' + ' < ' +
                inpfile + ' > ' + outfile + ' 2>&1\n')
        f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
        f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
        f.write('mv ' + temp_dir + prefix[index] + '.1drism ' + work_dir + prefix[index] + '.1drism\n')
        f.write('mv ' + temp_dir + prefix[index] + '.rism1 ' + work_dir + prefix[index] + '.rism1\n')
os.chmod(shfile, 0o744)

# %%
numcalc = 4
rvac_max = 25.0
rvac_min = 10.0
calclist = list(range(numcalc))
rvac = []
for i in calclist:
    rvac.append(rvac_min + rvac_max * i / (numcalc - i + 1))
lvac = 10.0 # 左側の真空の厚み
#rvac = 20.0 # 右側の真空の厚み
vac = (lvac + rvac)/2.0 # 表面構造関数には真空の左右の真空の厚みの半分を渡す
slab = fcc111('Al', size=(1,1,3), vacuum=vac)
slab.wrap() # 普通に作ると、ユニットセルをはみ出す原子があるので、ユニットセル内にwrapする。
slab.translate((0.0,0.0,(lvac-rvac)/2.0)) # lvac, rvacを反映した位置にずらす
#slab_sqrt3 = cut(slab, a=(3, 0, 0), b=(2, -4, 0), c=(0, 0, 1))
fig, ax = plt.subplots(1,2, figsize=(12, 6))
ax[0].set_axis_off()
ax[1].set_axis_off()
#plot_atoms(slab_sqrt3, ax[0], radii=1.2, rotation=('0x,0y,0z'))
#plot_atoms(slab_sqrt3, ax[1], radii=1.2, rotation=('90x,90y,90z'))
plot_atoms(slab, ax[0], radii=1.2, rotation=('0x,0y,0z'))
plot_atoms(slab, ax[1], radii=1.2, rotation=('90x,90y,90z'))
#view(slab, viewer='ngl')
plt.show()

# %%
#slab_ESM = copy.deepcopy(slab_sqrt3)
slab_ESM = copy.deepcopy(slab)
slab_ESM.translate((0.0,0.0,-slab.cell[2,2]/2.0)) # shift atoms to fit ESM/ESM-RISM model
starting_right = (slab_ESM.positions[1,2] + slab_ESM.positions[2,2])/2/Bohr

# %% tags=[]
pseudopotentials = {'Al':'Al.pbe-n-van.UPF'}
input_data = {
    'control': {
        'calculation': 'scf',
        'restart_mode': 'from_scratch',
        'pseudo_dir': '/Users/otani/Program/q-e/pseudo/',
        'outdir': '/Users/otani/Program/q-e/tempdir/',
        'prefix': 'Al100_bc1',
        'disk_io': 'low',
        'tprnfor': False,
    },
    'system': {
        'ibrav': 0,
        'ecutwfc': 25,
        'ecutrho': 200,
        'occupations' : 'smearing',
        'smearing':'gauss',
        'degauss' : 0.01,
        'tot_charge' : 0.0,
        'assume_isolated': 'esm',
        'esm_bc': 'bc1',
    },
    'electrons': {
        #'diagonalization': 'rmm-davidson',
        'diagonalization': 'rmm',
        'mixing_beta': 0.2,
    },
}

prefix = 'cell20'
inpfile = prefix + '.in'
write(inpfile, slab_ESM, format='espresso-in', input_data=input_data, 
      pseudopotentials=pseudopotentials,
      kpts=(8, 8, 1), koffset=(0, 0, 0))
outfile = prefix + '.out'
shfile = prefix + '.sh'
with open(shfile, 'w', encoding='UTF-8') as f:
    f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ../bin/pw.x -nk 4' + ' < ' + inpfile + ' > ' + outfile + ' 2>&1\n')
    f.write('mv ' + input_data['control'].get('prefix') + '.xml ' + prefix + '.xml\n')
    f.write('mv ' + input_data['control'].get('prefix') + '.esm1 ' + prefix + '.esm1\n')

# %%
print(32.67653/Bohr/2)

# %%
pseudopotentials = {'Al':'Al.pbe-n-van.UPF'}
input_data = {
    'control': {
        'calculation': 'scf',
        'restart_mode': 'from_scratch',
        'pseudo_dir': '/Users/otani/Program/q-e/pseudo/',
        'outdir': '/Users/otani/Program/q-e/tempdir/',
        'prefix': 'Al100_bc1',
        'disk_io': 'low',
        'trism': True,
        'tprnfor': False,
    },
    'system': {
        'ibrav': 0,
        'ecutwfc': 25,
        'ecutrho': 200,
        'occupations' : 'smearing',
        'smearing':'gauss',
        'degauss' : 0.01,
        'tot_charge' : 0.0,
        'assume_isolated': 'esm',
        'esm_bc': 'bc1',
    },
    'electrons': {
        #'diagonalization': 'rmm-davidson',
        'diagonalization': 'rmm',
        'mixing_beta': 0.2,
    },
    'rism': {
        'nsolv': 3,
        'closure': 'kh',
        'tempv': 300.0,
        'ecutsolv': 160.0,
        # 1D-RISM
        #'mdiis1d_size': 20,
        #'mdiis1d_step': 0.1,
        'rism1d_maxstep': 10000,
        #'rism1d_nproc': 36,
        #'rism1d_ngrid': 20000,
        #'rmax1d': 1500,
        'starting1d': 'file',
        # Lennard-Jones
        'solute_lj(1)': 'uff',
        # Laue-RISM
        #'rmax_lj': 40.0,
        'rism3d_conv_level': 0.4,
        'rism3d_maxstep': 1000,
        'laue_expand_right':60.0,
        'laue_starting_right': starting_right,
        'laue_rism_length_unit': 'angstrom',
        'laue_rism_length_right': 00.0,
    }
}
 
solv_info={
    'density_unit' : 'mol/L',
    'H2O':[-1,   'H2O.spc.MOL'],
    'Na+':[5.00, 'Na+.aq.MOL'],
    'Cl-':[5.00, 'Cl-.aq.MOL']
} 

prefix = 'cell20_ex60_ri00'
inpfile = prefix + '.in'
write(inpfile, slab_ESM, format='espresso-in', input_data=input_data, 
      pseudopotentials=pseudopotentials, solvents_info=solv_info,
      kpts=(4, 4, 1), koffset=(1, 1, 0))
outfile = prefix + '.out'
shfile = prefix + '.sh'
with open(shfile, 'w', encoding='UTF-8') as f:
    f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ../bin/pw.x -nk 4' + ' < ' + inpfile + ' > ' + outfile + ' 2>&1\n')
    f.write('mv ' + input_data['control'].get('prefix') + '.xml ' + prefix + '.xml\n')
    f.write('mv ' + input_data['control'].get('prefix') + '.esm1 ' + prefix + '.esm1\n')
    f.write('mv ' + input_data['control'].get('prefix') + '.rism1 ' + prefix + '.rism1\n')
    f.write('mv ' + input_data['control'].get('prefix') + '.1drism ' + prefix + '.1drism\n')

# %%
import qeschema

# %%
pw_document = qeschema.PwDocument()

# %%
pw_document.read("Al-NaCl_aq.xml")

# %%
xml_data = pw_document.to_dict()

# %% tags=[]
xml_data

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# # Dmitrii

# %%
work_dir = root_dir + 'Dmitrii/'

# %% [markdown]
# ### 構造を読み込む

# %%
qe_output = read_espresso_out(work_dir + '1c_slab322_vcrelax.log', -1)

# %%
aobj = qe_output
print(f' Unit cell: a = ({aobj.cell[0,0]:9.5f}, {aobj.cell[0,1]:9.5f}, {aobj.cell[0,2]:9.5f})')
print(f'            b = ({aobj.cell[1,0]:9.5f}, {aobj.cell[1,1]:9.5f}, {aobj.cell[1,2]:9.5f})')
print(f'            c = ({aobj.cell[2,0]:9.5f}, {aobj.cell[2,1]:9.5f}, {aobj.cell[2,2]:9.5f})')
print(f' Number of atoms: {len(aobj.positions):5d}')
print(f' Species, Positions:')
for i in range(len(aobj.positions)):
    print(f'  \'{aobj.symbols[i]:<2}\' ({aobj.positions[i,0]:9.5f}, {aobj.positions[i,1]:9.5f}, {aobj.positions[i,2]:9.5f})')

# %% [markdown] tags=[]
# rvac=16.0のセルでESM計算で収束した電子状態を使って本山さんの開発したコードでRISM領域を広げながら計算する。

# %%
prefix = ['r16+8_rism', 'r16+16_rism']
esm_prefix = 'r16_bc1'
lvac = 8.0
rvac = 16.0
work_dir = root_dir + '2022091115/'
if not os.path.exists(work_dir):
    print(work_dir + " not found. Make dir")
    os.makedirs(work_dir)
# QEbranch: rism_expand_oneside 54d66b07
qeroot_dir = '/Users/otani/Program/q-e/'
pseudo_dir = qeroot_dir + 'pseudo/'
temp_dir = qeroot_dir + 'tempdir/'

# %% tags=[]
rism_length = [8.0, 16.0]
for index, item in enumerate(prefix):
    #新しいobjectにqe_inputをコピー
    slab_ESM = copy.deepcopy(qe_input)
    #上の原子位置からスラブの厚みを単純に計算
    slab_thickness = qe_input.positions[2,2] - qe_input.positions[0,2]
    #スラブの左端の原子をz=0にずらす
    slab_ESM.translate((0.0,0.0,qe_input.positions[2,2]))
    #ユニットセルの大きさを再計算
    a = slab_ESM.cell[0,0]
    b = slab_ESM.cell[1,1]
    c = lvac + slab_thickness + rvac
    #ユニットセルの大きさを設定
    slab_ESM.set_cell([a,b,c])
    #ユニットセルの大きさの変更に合わせて、スラブの左端の原子を真空がlvacになるようにする
    slab_ESM.translate((0.0,0.0,lvac))
    #ESM用にずらす
    slab_ESM.translate((0.0,0.0,-slab_ESM.cell[2,2]/2.0))
    #PWの入力を作る
    pseudopotentials = {'Li':'Li.pbe-n-van.UPF'}
    input_data = {
        'control': {
            'calculation' : 'scf',
            'restart_mode': 'from_scratch',
            'pseudo_dir'  : pseudo_dir,
            'outdir'      : temp_dir,
            'prefix'      : prefix[index],
            'tprnfor'     : False,
            'trism'       : True,
        },
        'system': {
            'ibrav'          : 0,
            'ntyp'           : 0,
            'nat'            : 5,
            'ecutwfc'        : 50.0,
            'ecutrho'        : 300.0,
            'occupations'    : 'smearing',
            'smearing'       : 'gauss',
            'degauss'        : 0.01,
            'tot_charge'     : 0.0,
            'assume_isolated': 'esm',
            'esm_bc'         : 'bc1',
        },
        'electrons': {
            #'diagonalization': 'rmm-davidson',
            'diagonalization' : 'rmm',
            'mixing_beta'     : 0.4,
            'conv_thr'        : 1000.0,
            'startingpot'     : 'file',
            'startingwfc'     : 'file',
        },
        'rism': {
            'nsolv'                  : 1,
            'closure'                : 'kh',
            'tempv'                  : 300.0,
            'ecutsolv'               : 144.0,
            'solute_lj(1)'           : 'uff',
            'starting1d'             : 'zero',
            'starting3d'             : 'zero',
            'rism3d_conv_level'      : 0.7,
            'laue_rism_length_unit'  : 'angstrom',
            'laue_expand_right'      : 30.0,
            'laue_expand_left'       : 0.0,
            'laue_starting_right'    : -3.0,
            'laue_buffer_right'      : 8.0,
            'laue_rism_length_right' : rism_length[index],
        },
    }
    solv_info = {
        'density_unit' : 'mol/L',
        'EtOH' : [-1, 'Ethanol.oplsua.MOL'],
    }
    #1drismの結果があれば、それを読み込むように変更
    if os.path.isfile(temp_dir + prefix[index] + '.save/1d-rism_gvv_r.1.xml'):
        input_data['rism']['starting1d'] = 'file'
    #ESMの結果があれば、それを読み込むように変更
    if os.path.isfile(temp_dir + esm_prefix + '.save/charge-density.dat'):
        input_data['electrons']['startingpot'] = 'file'
        if not os.path.exists(temp_dir + prefix[index] + '.save'):
            print(temp_dir + prefix[index] + '.save' + " not found. Make dir")
            os.makedirs(temp_dir + prefix[index] + '.save')
        shutil.copy(temp_dir + esm_prefix + '.save/charge-density.dat', temp_dir + prefix[index] + '.save/')
    if os.path.isfile(temp_dir + esm_prefix + '.save/wfc1.dat'):
        input_data['electrons']['startingwfc'] = 'file'
        if not os.path.exists(temp_dir + prefix[index] + '.save'):
            print(temp_dir + prefix[index] + '.save' + " not found. Make dir")
            os.makedirs(temp_dir + prefix[index] + '.save')
        for file in glob.glob(temp_dir + esm_prefix + '.save/wfc*.dat'):
            shutil.copy(file, temp_dir + prefix[index] + '.save/')
    #inputファイルの出力
    inpfile = work_dir + prefix[index] + '.in'
    write(inpfile, slab_ESM, format='espresso-in', input_data=input_data, solvents_info=solv_info,
      pseudopotentials=pseudopotentials, kpts=(2, 2, 1), koffset=(0, 0, 0))
    #計算時の入出力ファイル
    inpfile = prefix[index] + '.in'
    outfile = prefix[index] + '.out'
    #逐次ジョブ投入用シェルスクリプト
    shfile = work_dir + prefix[index] + '.sh'
    with open(shfile, 'w', encoding='UTF-8') as f:
        f.write('#!/usr/bin/env bash\n')
        f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' +
                ' < ' + inpfile + ' > ' + outfile + ' 2>&1\n')
        f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
        f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
        f.write('mv ' + temp_dir + prefix[index] + '.1drism ' + work_dir + prefix[index] + '.1drism\n')
        f.write('mv ' + temp_dir + prefix[index] + '.rism1 ' + work_dir + prefix[index] + '.rism1\n')
    os.chmod(shfile, 0o744)
#一気にジョブ投入量シェルスクリプト
shfile = work_dir + 'all.sh'
with open(shfile, 'w', encoding='UTF-8') as f:
    f.write('#!/usr/bin/env bash\n')
    for index, item in enumerate(prefix):
        inpfile = prefix[index] + '.in'
        outfile = prefix[index] + '.out'
        f.write('export OMP_NUM_THREADS=1; mpirun -np 4 ' + qeroot_dir + 'bin/pw.x -nk 1' + ' < ' +
                inpfile + ' > ' + outfile + ' 2>&1\n')
        f.write('mv ' + temp_dir + prefix[index] + '.xml ' + work_dir + prefix[index] + '.xml\n')
        f.write('mv ' + temp_dir + prefix[index] + '.esm1 ' + work_dir + prefix[index] + '.esm1\n')
        f.write('mv ' + temp_dir + prefix[index] + '.1drism ' + work_dir + prefix[index] + '.1drism\n')
        f.write('mv ' + temp_dir + prefix[index] + '.rism1 ' + work_dir + prefix[index] + '.rism1\n')
os.chmod(shfile, 0o744)

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[] toc-hr-collapsed=true
# ### Li-EtOHの(conventional)ESM-RISM計算(DFT fix)

# %%
print(chr(0x03B1),chr(0x212B),'\u03B1')

# %%
greek_letterz=[chr(code) for code in range(945,970)]

print(greek_letterz)

# %%
print(r'$\alpha$')

# %%

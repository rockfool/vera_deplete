"""
problem 1A pin 3.1 w/o TF=565K
moderator 600 K 
clad      600 K
fuel      900 K
moderator density 0.700 g/cc
U-235  w/o 3.1
power density 40.0 W/gU
"""

import openmc
import openmc.deplete
import numpy as np
import matplotlib.pyplot as plt
import json 

###############################################################################
#                      Simulation Input File Parameters
###############################################################################

# OpenMC simulation parameters
batches   = 220
inactive  = 20
particles = 100000

# Depletion simulation parameters
time_step = 1*24*60*60 # s
final_time = 5*24*60*60 # s
#time_steps = np.full(final_time // time_step, time_step)
time_steps = np.array([0.25, 6.00, 6.25, 12.50, 25.00, 25.00, 25.00, # 100.00 EFPD 
             25.00, 25.00, 25.00, 25.00, 25.00, 25.00, 25.00, 25.00, # 300.00 EFPD
             25.00, 25.00, 25.00, 25.00, 25.00, 25.00, 25.00, 25.00, # 500.00 EFPD
             62.50, 62.50, 62.50, 62.50, 62.50, 62.50, 62.50, 62.50, # 1000.0 EFPD
             62.50, 62.50, 62.50, 62.50, 62.50, 62.50, 62.50, 62.50  # 1500.0 EFPD
             ])*24*60*60 

#chain_file = './chain_casl.xml'
power_den = 40.0 # W/cm, for 2D simulations only (use W for 3D) 40.0 W/gU 

###############################################################################
#                              Define materials
###############################################################################

# Instantiate some Materials and register the appropriate Nuclides
fuel_21 = openmc.Material(material_id=1, name='UO2 fuel at 2.1% wt enrichment')
fuel_21.set_density('g/cm3', 10.257)
fuel_21.add_nuclide('U234',4.02487E-06 )
fuel_21.add_nuclide('U235',4.86484E-04 )
fuel_21.add_nuclide('U236',2.23756E-06 )
fuel_21.add_nuclide('U238',2.23868E-02 )
fuel_21.add_nuclide('O16' ,4.57590E-02 )
fuel_21.depletable = True


fuel_31 = openmc.Material(material_id=2, name='UO2 fuel at 3.1% wt enrichment')
fuel_31.set_density('g/cm3', 10.257)
fuel_31.add_nuclide('U234',6.11864E-06 )
fuel_31.add_nuclide('U235',7.18132E-04 )
fuel_31.add_nuclide('U236',3.29861E-06 )
fuel_31.add_nuclide('U238',2.21546E-02 )
fuel_31.add_nuclide('O16' ,4.57642E-02 )
fuel_31.depletable = True


fuel_36 = openmc.Material(material_id=3, name='UO2 fuel at 3.6% wt enrichment')
fuel_36.set_density('g/cm3', 10.257)
fuel_36.add_nuclide('U234',7.21203E-06 )
fuel_36.add_nuclide('U235',8.33952E-04 )
fuel_36.add_nuclide('U236',3.82913E-06 )
fuel_36.add_nuclide('U238',2.20384E-02 )
fuel_36.add_nuclide('O16' ,4.57669E-02 )
fuel_36.depletable = True


fuel_46 = openmc.Material(material_id=4, name='UO2 fuel at 4.6% wt enrichment')
fuel_46.set_density('g/cm3', 10.257)
fuel_46.add_nuclide('U234',9.39876E-06 )
fuel_46.add_nuclide('U235',1.06559E-03 )
fuel_46.add_nuclide('U236',4.89014E-06 )
fuel_46.add_nuclide('U238',2.18062E-02 )
fuel_46.add_nuclide('O16' ,4.57721E-02 )
fuel_46.depletable = True


helium = openmc.Material(material_id=5, name='Helium for gap')
helium.set_density('g/cm3', 0.000179)
helium.add_nuclide('He4', 2.68714E-5)

zirc4 = openmc.Material(material_id=6, name='Zircaloy 4')
zirc4.set_density('g/cm3', 6.56)
zirc4.add_nuclide('Zr90',  2.18865E-02)
zirc4.add_nuclide('Zr91',  4.77292E-03)
zirc4.add_nuclide('Zr92',  7.29551E-03)
zirc4.add_nuclide('Zr94',  7.39335E-03)
zirc4.add_nuclide('Zr96',  1.19110E-03)
zirc4.add_nuclide('Sn112',  4.68066E-06)
zirc4.add_nuclide('Sn114',  3.18478E-06)
zirc4.add_nuclide('Sn115',  1.64064E-06)
zirc4.add_nuclide('Sn116',  7.01616E-05)
zirc4.add_nuclide('Sn117',  3.70592E-05)
zirc4.add_nuclide('Sn118',  1.16872E-04)
zirc4.add_nuclide('Sn119',  4.14504E-05)
zirc4.add_nuclide('Sn120',  1.57212E-04)
zirc4.add_nuclide('Sn122',  2.23417E-05)
zirc4.add_nuclide('Sn124',  2.79392E-05)
zirc4.add_nuclide('Fe54',  8.68307E-06)
zirc4.add_nuclide('Fe56',  1.36306E-04)
zirc4.add_nuclide('Fe57',  1.36306E-04)
zirc4.add_nuclide('Fe58',  4.18926E-07)
zirc4.add_nuclide('Cr50',  3.30121E-06)
zirc4.add_nuclide('Cr52',  6.36606E-05)
zirc4.add_nuclide('Cr53',  7.21860E-06)
zirc4.add_nuclide('Cr54',  1.79686E-06)
zirc4.add_nuclide('Hf174',  3.54138E-09)
zirc4.add_nuclide('Hf176',  1.16423E-07)
zirc4.add_nuclide('Hf177',  4.11686E-07)
zirc4.add_nuclide('Hf178',  6.03806E-07)
zirc4.add_nuclide('Hf179',  3.01460E-07)
zirc4.add_nuclide('Hf180',  7.76449E-07)

water_565 = openmc.Material(material_id=7, name='Borated water at 565 K with 1300 ppm')
water_565.set_density('g/cm3', 0.743)
water_565.add_nuclide('O16',2.48112E-02 )
water_565.add_nuclide('H1' ,4.96224E-02 )
water_565.add_nuclide('B10',1.07070E-05 )
water_565.add_nuclide('B11',4.30971E-05 )
water_565.add_s_alpha_beta('c_H_in_H2O')


water_600= openmc.Material(material_id=8, name='Borated water at 600 K with 1300 ppm')
water_600.set_density('g/cm3', 0.700)
water_600.add_nuclide('O16', 2.33753E-02)
water_600.add_nuclide('H1', 4.67505E-02)
water_600.add_nuclide('B10', 1.00874E-05)
water_600.add_nuclide('B11', 4.06030E-05)
water_600.add_s_alpha_beta('c_H_in_H2O')


ifba= openmc.Material(material_id=9, name='IFBA')
ifba.set_density('g/cm3', 3.85)
ifba.add_nuclide('B10',  2.16410E-02)
ifba.add_nuclide('B11',  1.96824E-02)
ifba.add_nuclide('Zr90', 1.06304E-02)
ifba.add_nuclide('Zr91', 2.31824E-03)
ifba.add_nuclide('Zr92', 3.54348E-03)
ifba.add_nuclide('Zr94', 3.59100E-03)
ifba.add_nuclide('Zr96', 5.78528E-04)


pyrex= openmc.Material(material_id=10, name='Pyrex')
pyrex.set_density('g/cm3', 2.25)
pyrex.add_nuclide('B10', 9.63266E-04) 
pyrex.add_nuclide('B11', 3.90172E-03)
pyrex.add_nuclide('O16', 4.67761E-02)
pyrex.add_nuclide('Si28', 1.81980E-02)
pyrex.add_nuclide('Si29', 9.24474E-04)
pyrex.add_nuclide('Si30', 6.10133E-04)


gad= openmc.Material(material_id=11, name='gad with 5% Gd2O3 95% UO2, 1.8 % U-235')
gad.set_density('g/cm3', 10.111)
gad.add_nuclide('U234', 3.18096E-06)
gad.add_nuclide('U235', 3.90500E-04)
gad.add_nuclide('U236', 1.79300E-06)
gad.add_nuclide('U238', 2.10299E-02)
gad.add_nuclide('Gd152', 3.35960E-06)
gad.add_nuclide('Gd154', 3.66190E-05)
gad.add_nuclide('Gd155', 2.48606E-04)
gad.add_nuclide('Gd156', 3.43849E-04)
gad.add_nuclide('Gd157', 2.62884E-04)
gad.add_nuclide('Gd158', 4.17255E-04)
gad.add_nuclide('Gd160', 3.67198E-04)
gad.add_nuclide('O16'  , 4.53705E-02)
gad.depletable = True


ss304= openmc.Material(material_id=12, name='SS304')
ss304.set_density('g/cm3', 7.8)
ss304.add_element('C'  ,  3.20895E-04)
ss304.add_nuclide('Si28', 1.58197E-03)
ss304.add_nuclide('Si29', 8.03653E-05)
ss304.add_nuclide('Si30', 5.30394E-05)
ss304.add_nuclide('P31',  6.99938E-05)
ss304.add_nuclide('Cr50', 7.64915E-04)
ss304.add_nuclide('Cr52', 1.47506E-02)
ss304.add_nuclide('Cr53', 1.67260E-03)
ss304.add_nuclide('Cr54', 4.16346E-04)
ss304.add_nuclide('Mn55', 1.75387E-03)
ss304.add_nuclide('Fe54', 3.44776E-03)
ss304.add_nuclide('Fe56', 5.41225E-02)
ss304.add_nuclide('Fe57', 1.24992E-03)
ss304.add_nuclide('Fe58', 1.66342E-04)
ss304.add_nuclide('Ni58', 5.30854E-03)
ss304.add_nuclide('Ni60', 2.04484E-03)
ss304.add_nuclide('Ni61', 8.88879E-05)
ss304.add_nuclide('Ni62', 2.83413E-04)
ss304.add_nuclide('Ni64', 7.21770E-05)


agincd= openmc.Material(material_id=13, name='Ag-In-Cd')
agincd.set_density('g/cm3', 10.2)
agincd.add_nuclide('Ag107', 2.36159E-02)
agincd.add_nuclide('Ag109', 2.19403E-02)
agincd.add_nuclide('Cd106', 3.41523E-05)
agincd.add_nuclide('Cd108', 2.43165E-05)
agincd.add_nuclide('Cd110', 3.41250E-04)
agincd.add_nuclide('Cd111', 3.49720E-04)
agincd.add_nuclide('Cd112', 6.59276E-04)
agincd.add_nuclide('Cd113', 3.33873E-04)
agincd.add_nuclide('Cd114', 7.84957E-04)
agincd.add_nuclide('Cd116', 2.04641E-04)
agincd.add_nuclide('In113', 3.44262E-04)
agincd.add_nuclide('In115', 7.68050E-03)

b4c= openmc.Material(material_id=14, name='B4C')
b4c.set_density('g/cm3', 1.76)
b4c.add_nuclide('B10',1.52689E-02)
b4c.add_nuclide('B11',6.14591E-02)
b4c.add_element('C'  ,1.91820E-02)

waba= openmc.Material(material_id=15, name='WABA B4C-Al2O3')
waba.set_density('g/cm3', 3.65)
waba.add_nuclide('B10', 2.98553E-03)
waba.add_nuclide('B11', 1.21192E-02)
waba.add_element('C'  , 3.77001E-03)
waba.add_nuclide('O16', 5.85563E-02)
waba.add_nuclide('Al27',3.90223E-02)

materials_file = openmc.Materials([fuel_21,fuel_31, fuel_36, fuel_46,helium, water_565, water_600,pyrex,zirc4,gad,ifba,ss304 ,agincd,b4c,waba])
materials_file.export_to_xml()
###############################################################################
#                             Create geometry
###############################################################################

# Instantiate zCylinder surfaces
fuel_or = openmc.ZCylinder(surface_id=1, x0=0, y0=0, r=0.4096, name='Fuel OR')
clad_ir = openmc.ZCylinder(surface_id=2, x0=0, y0=0, r=0.4180, name='Clad IR')
clad_or = openmc.ZCylinder(surface_id=3, x0=0, y0=0, r=0.4750, name='Clad OR')
gt_ir   = openmc.ZCylinder(surface_id=4, x0=0, y0=0, r=0.5610, name='Guide Tube IR')
gt_or   = openmc.ZCylinder(surface_id=5, x0=0, y0=0, r=0.6020, name='Guide Tube OR')
it_ir   = openmc.ZCylinder(surface_id=6, x0=0, y0=0, r=0.5590, name='Instr Tube IR')
it_or   = openmc.ZCylinder(surface_id=7, x0=0, y0=0, r=0.6050, name='Instr Tube OR')
th_ir   = openmc.ZCylinder(surface_id=8, x0=0, y0=0, r=0.2580, name='Thimble IR')
th_or   = openmc.ZCylinder(surface_id=9, x0=0, y0=0, r=0.3820, name='Thimble OR')
#
pin_left   = openmc.XPlane(surface_id=11,  x0=-0.6300, name='left')
pin_right  = openmc.XPlane(surface_id=12,  x0= 0.6300, name='right')
pin_back   = openmc.YPlane(surface_id=13,  y0=-0.6300, name='back')
pin_front  = openmc.YPlane(surface_id=14,  y0= 0.6300, name='front')
pin_bottom = openmc.ZPlane(surface_id=15,  z0=-200.0, name='bottom')
pin_top    = openmc.ZPlane(surface_id=16,  z0= 200.0, name='top')

pin_left.boundary_type   = 'reflective'
pin_right.boundary_type  = 'reflective'
pin_back.boundary_type   = 'reflective'
pin_front.boundary_type  = 'reflective'
pin_bottom.boundary_type = 'reflective'
pin_top.boundary_type    = 'reflective'
#
assembly_left   = openmc.XPlane(surface_id=21,  x0= 10.75, name='left')
assembly_right  = openmc.XPlane(surface_id=22,  x0= 10.75, name='right')
assembly_back   = openmc.YPlane(surface_id=23,  y0=-10.75, name='back')
assembly_front  = openmc.YPlane(surface_id=24,  y0= 10.75, name='front')
assembly_bottom = openmc.ZPlane(surface_id=25,  z0=-200.0, name='bottom')
assembly_top    = openmc.ZPlane(surface_id=26,  z0= 200.0, name='top')
#
assembly_left.boundary_type   = 'reflective'
assembly_right.boundary_type  = 'reflective'
assembly_back.boundary_type   = 'reflective'
assembly_front.boundary_type  = 'reflective'
assembly_bottom.boundary_type = 'reflective'
assembly_top.boundary_type    = 'reflective'

# Instantiate cells
fuel  = openmc.Cell(cell_id=1, name='cell 1')
gap   = openmc.Cell(cell_id=2, name='cell 2')
clad  = openmc.Cell(cell_id=3, name='cell 3')
water = openmc.Cell(cell_id=4, name='cell 4')

# Use surface half-spaces to define regions
fuel.region = -fuel_or
gap.region = +fuel_or & -clad_ir
clad.region = +clad_ir & -clad_or
water.region = +clad_or & +pin_left & -pin_right & +pin_back & -pin_front

# Register materials with Cells
fuel.fill = fuel_31
gap.fill  = helium 
clad.fill = zirc4
water.fill = water_600 
# temperature definition
fuel.temperature  = 900.0
gap.temperature   = 600.0
clad.temperature  = 600.0
water.temperature = 600.0

# Instantiate universe
root = openmc.Universe(universe_id=0, name='root universe')

# Register cells with Universe
root.add_cells([fuel, gap, clad, water])

# Instantiate a Geometry, register the root Universe
geometry = openmc.Geometry(root)

geometry.export_to_xml()

###############################################################################
#                     Transport calculation settings
###############################################################################

# Instantiate a Settings object, set all runtime parameters, and export to XML
settings_file = openmc.Settings()
settings_file.batches = batches
settings_file.inactive = inactive
settings_file.particles = particles
settings_file.temperature={
		 'method'    :'nearest',
		 'tolerance' : 300.0,
		 'multipole' : False
 }

settings_file.resonance_scattering = {
                 'enable'    : False,
                 'method'    : 'dbrc'
}
# Create an initial uniform spatial source distribution over fissionable zones
bounds = [-0.6300, -0.6300, -1, 0.6300, 0.6300, 1]
uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:], only_fissionable=True)
settings_file.source = openmc.source.Source(space=uniform_dist)

entropy_mesh = openmc.RegularMesh()
entropy_mesh.lower_left  = [-0.4096, -0.4096, -1.e50]
entropy_mesh.upper_right = [ 0.4096,  0.4096,  1.e50]
entropy_mesh.dimension = [10, 10, 1]
settings_file.entropy_mesh = entropy_mesh

settings_file.export_to_xml()

# plot setting 
plot = openmc.Plot(plot_id=1)
plot.origin = [0, 0, 0]
plot.width  = [1.26, 1.26]
plot.pixels = [500, 500]
plot.color_by = 'material'
# Instantiate a Plots object and export to XML
plot_file = openmc.Plots([plot])
plot_file.export_to_xml()


###############################################################################
#                     Set volumes of depletable materials
###############################################################################

# Compute cell areas
area = {}
area[fuel] = np.pi * fuel_or.coefficients['r'] ** 2

# Set materials volume for depletion. Set to an area for 2D simulations
fuel_31.volume = area[fuel]


###############################################################################
#                   Initialize and run depletion calculation
###############################################################################

# op = openmc.deplete.Operator(geometry, settings_file, chain_file, diff_burnable_mats=True)

# Perform simulation using the predictor algorithm
# openmc.deplete.integrator.cecm(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.predictor(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.cecm(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.epc_rk4(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.leqi(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.celi(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.si_celi(op, time_steps, power_density=power_den)
# openmc.deplete.integrator.si_leqi(op, time_steps, power_density=power_den)

# Get fission Q values from JSON file generated by get_fission_qvals.py
with open('/home/jiankai/depletion-comparison/data/depletion/serpent_fissq.json', 'r') as f:
    serpent_fission_q = json.load(f)

# Set up depletion operator
chain_file = '/home/jiankai/depletion-comparison/data/depletion/chain_casl_pwr.xml'
op = openmc.deplete.Operator(geometry, settings_file, chain_file, diff_burnable_mats=True, 
    fission_q=serpent_fission_q,
    fission_yield_mode="average")
    
integrator = openmc.deplete.CELIIntegrator(op, time_steps, power_density=power_den)

integrator.integrate()


import pds4_tools
# pds4_tools.view()
# pds4_tools.view('ch2_ohr_ncp_20240425T1603031918_d_img_d18.xml')

structures = pds4_tools.read('ch2_ohr_ncp_20240425T1603031918_d_img_d18.xml')
print(structures)
# pds4_tools.view(from_existing_structures=structures)

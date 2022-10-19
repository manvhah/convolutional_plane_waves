FIGS = ./figures

all: genfigs

.PHONY: install data genfigs

install:
	pip install git+https://github.com/manvhah/sporco.git@feature/smooth_csc_sporco020
	pip install wheel numpy matplotlib scipy sklearn sporco cvxpy pebble h5py pandas

data:
ifneq ("$(wildcard ./data/classroom_frequency_responses.h5)","")
	echo "./data/classroom_frequency_responses.h5 exists already."
else
	cd data
	@echo " > Downloading data, approx 364 MB .."
	wget --no-check-certificate -O ./data/classroom_frequency_responses.h5 https://data.dtu.dk/ndownloader/files/27505451
	cd ..
endif

genfigs: 
	@echo " > Generating figures ..."
	@mkdir -p $(FIGS)
	@touch $(FIGS)/dummy
	@rm $(FIGS)/*
	@echo " > Fig1 ..."
	python3 partitioning.py
	@echo " > Fig 2+3 ..."
	python3 csc1d_radial.py
	@echo " > Fig 4+7 ..."
	python3 gen_figures.py
	@echo " > Fig 5 ..."
	python3 test_sfr.py
	@echo " > Fig 8 ..."
	python3 read_mc.py global localindep conv -pdfname conv_paper -legends
	@echo " > rename figures according to paper ..."
	@mv $(FIGS)/partitioning.pdf                          $(FIGS)/Figure1.pdf
	@mv $(FIGS)/rec_1dcsc.pdf                             $(FIGS)/Figure2.pdf
	@mv $(FIGS)/nmse_distance_monopole.pdf                $(FIGS)/Figure3.pdf
	@mv $(FIGS)/rec_monoplane_343_gpwe_lpwe_cpwe_cpwe.pdf $(FIGS)/Figure4.pdf
	@mv $(FIGS)/particle_velocity_gpwe_monoplane.pdf      $(FIGS)/Figure5a.pdf
	@mv $(FIGS)/particle_velocity_lpwe_monoplane.pdf      $(FIGS)/Figure5b.pdf
	@mv $(FIGS)/particle_velocity_cpwe_monoplane.pdf      $(FIGS)/Figure5c.pdf
	@mv $(FIGS)/classroom.pdf                             $(FIGS)/Figure6.pdf
	@mv $(FIGS)/rec_019_1000_gpwe_lpwe_cpwe_cpwe.pdf      $(FIGS)/Figure7.pdf
	@mv $(FIGS)/csc_paper_xfreq_mic_nmse.pdf              $(FIGS)/Figure8.pdf
	@echo " > ... done"

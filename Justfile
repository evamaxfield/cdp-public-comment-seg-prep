# list all available commands
default:
  just --list

###############################################################################
# Basic project and env management

# clean all build, python, and lint files
clean:
	rm -fr dist
	rm -fr .eggs
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .mypy_cache

# install with all deps
install:
	pip install uv
	uv pip install -r requirements.txt
	uv pip install -r requirements-dev.txt

# lint, format, type check
lint:
	pre-commit run --all-files

###############################################################################
# Infra / Data Storage

# Default region for infrastructures
key_guid := replace_regex(uuid(), "([a-z0-9]{1})(.*)", "$1")
default_key := justfile_directory() + "/.keys/dev.json"
default_project := "cdp-public-comment-data-proc"
default_bucket := "gs://cdp-public-comment-data-proc"

# run gcloud login
login:
  gcloud auth login
  gcloud auth application-default login

# generate a service account JSON
gen-key project=default_project:
	mkdir -p {{justfile_directory()}}/.keys/
	gcloud iam service-accounts create {{project}}-{{key_guid}} \
		--description="Dev Service Account {{key_guid}}" \
		--display-name="{{project}}-{{key_guid}}"
	gcloud projects add-iam-policy-binding {{project}} \
		--member="serviceAccount:{{project}}-{{key_guid}}@{{project}}.iam.gserviceaccount.com" \
		--role="roles/owner"
	gcloud iam service-accounts keys create {{justfile_directory()}}/.keys/{{project}}-{{key_guid}}.json \
		--iam-account "{{project}}-{{key_guid}}@{{project}}.iam.gserviceaccount.com"
	cp -rf {{justfile_directory()}}/.keys/{{project}}-{{key_guid}}.json {{default_key}}
	@ echo "----------------------------------------------------------------------------"
	@ echo "Sleeping for thirty seconds while resources set up"
	@ echo "----------------------------------------------------------------------------"
	sleep 30
	@ echo "Be sure to update the GOOGLE_APPLICATION_CREDENTIALS environment variable."
	@ echo "----------------------------------------------------------------------------"

# create a new gcloud project and generate a key
init project=default_project:
	gcloud projects create {{project}} --set-as-default
	echo "----------------------------------------------------------------------------"
	echo "Follow the link to setup billing for the created GCloud account."
	echo "https://console.cloud.google.com/billing/linkedaccount?project={{project}}"
	echo "----------------------------------------------------------------------------"
	just gen-key {{project}}

# switch active gcloud project
switch-project project=default_project:
	gcloud config set project {{project}}

# enable gcloud services
enable-services project=default_project:
	gcloud services enable cloudresourcemanager.googleapis.com
	gcloud services enable \
		storage.googleapis.com \
		compute.googleapis.com \
		logging.googleapis.com \
		monitoring.googleapis.com

# setup all resources
setup-infra project=default_project:
	just enable-services {{project}}
	-gcloud storage buckets create {{default_bucket}}
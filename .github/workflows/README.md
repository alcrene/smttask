
# GitHub to PyPI in 30 minutes

The PyPI is a godsend for Python users, placing most packages they need an easy `pip install` away.
The process of *putting* code on PyPI however has long been much more convoluted.

The good news is that the last few years have seen massive standardization efforts on the Python packaging front, meaning that for simple projects, it is now possible to devise an (almost) completely portable procedure that will work for all of them.
The less good news is that with packaging tools still in a bit of flux, a lot of the information you find on the web may seem contradictory, and most of it makes the process more complicated than it now needs to be.

This little resource is my attempt at collating current recommendations, as of August 2023, into the simplest possible path to PyPI publication. It provides workflow files & guidelines that I can reuse across multiple projects. My hope is that they will also be useful to you, either as⁻is or as a starting point for your own packaging adventure.

This is **not** a tutorial on publishing to PyPI. That would be beside the point, since it would make this short document very much not short. Also, there are already multiple tutorials available, which are more likely to stay up to date than whatever I write here. My goal here is not for completeness, but maximum concision. If this is your first time using these tools, you will need to look up how they work separately.

On the other hand, if you just want to get your package on PyPI as quickly as possible and with minimum fuss, this is the resource for you. Just copy a few files into your repo, add the project to PyPI, and you should be good to go. 

## Assumptions & tool choices

There are different ways to get a package on PyPI. This guidebook assumes or uses the following:

- [PyPA’s `build`](https://pypa-build.readthedocs.io/en/latest/)
    + This means that only [`pyproject.toml`](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) projects are supported.
- Version numbers are automatically determined by [`setuptools-scm`](https://github.com/pypa/setuptools_scm), with [SemVer](https://semver.org/) tag names.
    + In particular, this means the `pyproject.toml` should not specify the version explicitely, but rather contain a line like `dynamic = ["version"]`.
- [Trusted publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/) to avoid the need for API tokens.
- GitHub, because it is currently the only Trusted Publisher.
- [GitHub Actions](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
    + Once properly configured, this allows publishing literally at the click of a button. (In contrast to e.g. `twine`, which requires issuing commands you probably forgot since the last time.)
- Separate GitHub workflows for building and publishing.
    + [pypi-publish](https://github.com/marketplace/actions/pypi-publish) action
    + [dawidd6/action-download-artifact](https://github.com/dawidd6/action-download-artifact) in order to separate build and deployment into separate workflows.

## Step-by-step instructions for publishing

### Create accounts

(Skip if you already have accounts on these services.)

- On [TestPyPI](https://test.pypi.org)
- On [PyPI](https://pypi.org/)

### Repo configuration

Do this with the GitHub web UI. (These are my personal preferences; adapt as desired.)

- Settings
    - General
        - Default branch
            - main
    - Branches
        - Add a rule for branch `main`
            - Allow force push
                - Only admin
    - Tags
        - Protected tags
            - Add the rule `v*`
    - Environments
        - Create 2 environments
            - `release`
                - Wait timer: 15 minutes[^why-timer]
                - Limit to protected branches
            - `release-testpypi`
                - Limit to protected branches

[^why-timer]: Once a package is published to PyPI, it cannot be replaced except by publishing a new version with a newer version number. The timer is intended to give you time to realize a mistake and cancel the `release` action before it runs.

### Add GitHub workflows

- Copy the three GitHub workflow files from this repo to your project repo under `.github/workflows/`
    + There are different ways to do this. One way is to clone the repo with [git-subrepo](https://github.com/ingydotnet/git-subrepo):

          mkdir .github
          git subrepo clone https://github.com/alcrene/publish-to-pypi .github/workflows

- Edit `.github/workflows/build.yml` as needed.
    + In particular, check that the Python version is appropriate.
    
#### Description of the workflows

- `build.yml`:
    + Will build the distribution files (runs `python3 -m build`)
    + Will run:
        + when a new release is created with the GitHub UI;
        + when triggered manually from the *Actions* menu.

- `publish-on-pypi.yml`:
    + Will place the package on the official PyPI index
    + Permanent: Once a package is published with PyPI with a version number, no new packages can be published with the same number.
    + Runs within the environment `release`.
    + Will run:
        + when triggered manually from the *Actions* menu.

- `publish-on-testpypi.yml`:
    + Will place the package on the test PyPI index
    + The test index can be used
        + to make sure the packaging pipeline works as it should;
        + to share pre-release with others to allow them to test them.
            + Packages on test PyPI are installed with `pip install -i https://test.pypi.org/simple/ <pkg name>`
    + Runs within the environment `release-testpypi`.
    + Will run:
        + when triggered manually from the *Actions* menu
                
### Prepare PyPI/TestPyPI

Actions are configured to use [*Trusted publishing*](https://docs.pypi.org/trusted-publishers/using-a-publisher/). This avoids the need to generate and store API tokens in the job file, and thus enables (almost) completely generic jobs. What we need to do however is tell PyPI / TestPyPI to expect our new package

From your PyPI user page:

* Publication
    + Add a new pending publisher
        - Project name: must match the `name` field in `pyproject.toml`
        - Workflow name: `publish-on-pypi.yml`
        - Environment name: `release`
    
If you intend to use TestPyPI, you need to repeat the procedure there

* Workflow name: `publish-on-testpypi.yml`
* Environment name: `release-testpypi`

### Publish a release candidate to TestPyPI

When you are ready to publish a new release, do the following:
(The procedure is the same for the first or subsequent releases.)

- Tag the latest commit with an RC version number: `v0.1.0-rc.1`
  + Make sure to use a [SemVer](https://semver.org/) so that build tools recognize the version number.
  + The GitHub actions will always build and publish the latest commit with a SemVer tag.
  + Versions prefixed with `v` is the most common standard, and some tools may expect it.
  + Separating numbers – like `-rc.1` – ensures that they order properly with versions like `-rc.11`.
  + `setuptools_scm` will parse the version from the tag name and produce a nice number for the package: `0.1.0-rc1`.
  
- Push the tag to GitHub

- Trigger the `build.yml` action manually.
  + Alternatively, make a new release with the GitHub UI.  
    (Unfortunately, there seems to be an issue with automatically triggered builds. See below.)
  
- If the build looks OK, trigger the `publish-on-testpypi.yml` manually from the Actions menu.

- Test as needed.

### Publish a new official version on PyPI

- Tag the latest commit with plain version number: `v0.1.0`
  + Make sure to use a [SemVer](https://semver.org/) so that build tools recognize the version number.
  + This *must* be the latest commit with a SemVer tag.
  + This *may* be the same commit as the latest RC version (and probably should), so the commit will have both an RC version tag and a non-RC version tag.
  + Versions prefixed with `v` is the most common standard, and some tools may expect it.
  
- Push the tag to GitHub

- Make a new release with the GitHub UI.
  + This should trigger a new build.  
    (Unfortunately, for reasons I don’t understand, these build artifacts will not be found by the publish actions. These seems to only happen when the build action is automatically triggered by a release.)

- WORKAROUND: Manually trigger the build action.
  
- If the build looks OK, trigger the `publish-on-pypi.yml` manually from the Actions menu.
  + This will wait 15 minutes before starting. If during this window you realize you forgot something, you may cancel the action from the Action menu.

- Your package is now on PyPI !

## Attribution

The GitHub actions are derived from examples from the [PyPI docs](https://docs.pypi.org/trusted-publishers/using-a-publisher/) (Apache-2.0) and the [PyPI publish GitHub Action](https://github.com/marketplace/actions/pypi-publish#usage) (BSD-3).

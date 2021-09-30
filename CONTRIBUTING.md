# Contributors

## Setup

1. Fork the repository by clicking on the 'Fork' button on the repository's page. This creates a copy of the code under your GitHub user account.
1. Connect to the university vpn to be able to access the FDN-Server. [Tutorial](https://www.unibe.ch/university/campus_and_infrastructure/rund_um_computer/internetzugang/access_to_internal_resources_via_vpn/index_eng.html)
1. Connect to the FDN Server via ssh with the command `ssh <USERNAME>@fnd-sandbox3.inf.unibe.ch`.
1. In your homedirectory (default location after first login) you can clone your repository and then add the base repository as a remote
    ```bash
    git clone git@github.com:<YOURGITHUBHANDLE>/SwissCourtRulingCorpus.git
    cd SwissCourtRulingCorpus
    git remote add upstream https://github.com/JoelNiklaus/SwissCourtRulingCorpus.git
    ``` 
1. Create a new branch with a descriptive change of your work, where you will make your changes:
    ```
    git checkout -b a-descriptive-name-for-my-changes
    ```


## Developing

### Before starting
After logging in you should be in the conda `scrc` environment which is visible in your terminal 
    ```bash
    (scrc) <USERNAME>@fdn-sandbox3:~$
    ``` 
You can find the code in the cloned GitHub directory `SwissCourtRulingCorpus`.


### Coding

Make the changes in the files you want to modify or create files accordingly. To get an idea how you might add something, don't hesitate to explore the code.

If you have edited the code, you can run it via the following command 
    ```bash
    python -m PATH.TO.MODULE
    ```
where the path to the module for example is `scrc.main` or `scrc.dataset_construction.section_splitter`

If you're happy with your changes add your files and make a commit to record them locally:
    ```
    git add <FILE YOU MODIFIED>
    git commit
    ```
For good commit messages use for example the following guide: https://chris.beams.io/posts/git-commit/

It is a good idea to sync your copy of the code with the original repository regularly. This way you can quickly account for changes:
    ```bash
    git fetch upstream
    git rebase upstream/master
    ```


### Pushing your code

Push the changes to your account using:

    ```bash
    git push -u origin a-descriptive-name-for-my-changes
    ```

Then you can go to to your GitHub Repository and click on Pull Request, where your codes gets reviewed.

## Questions?

Do not hesitate to contact Adrian on Slack or via email: adrian.joerg@inf.unibe.ch
This is webapp i developed 


# master_thesis_deploy

#how to run this app .

fork this repo 

upload file in aws in s3  and change file directory(replace aws folder name of mine with yours) bucket_name in all.py python file which has main function.

create aws key and code using IAM role

go to streamlit . 

selected python version 3.9 in strealit while deploying (in general setting tab)

add aws key and id in secrets configuration of app.

deploy it using sreamlit app .

https://share.streamlit.io/


copy paste aws key here 

https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html

The streamlit will download all the library as mentioned in requirements.txt

and then explore the webapp 

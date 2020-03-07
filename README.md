# LooC-web
Web interface for the LooC project

## Set up
#### Clone this repo
`git clone git@github.com:ElementAI/looc-web.git`

#### Instantiate web server
`$ cd looc-web`
`gunicorn -w 2 -b <IP> wsgi:app`

pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "automated-lodge-476410-h0"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
    }

    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins............'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/atul219/Hotel_Reservation.git']])
                }
            }
        }

        stage('Setting up out virtual environment and installing dependecies'){
            steps{
                script{
                    echo 'Setting up out virtual environment and installing dependecies'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }

        stage('Building and Pushing Docker Image to GCR'){
            steps{
                withCredentials([file(credentialsId: 'gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo 'Building and Pushing Docker Image to GCR.............'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}


                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

                        gcloud config set project ${GCP_PROJECT}

                        gcloud auth configure-docker --quiet

                        # Ensure buildx is active (no-op if already created)
                        docker buildx create --use || true

                        # Build with secret; push (or use --load + docker push if you prefer)
                        docker buildx build \
                            --secret id=gcp_key,src="${GOOGLE_APPLICATION_CREDENTIALS}" \
                            -t gcr.io/${GCP_PROJECT}/ml-project:latest \
                            --push \
                            .

                        '''
                    }
                }
            }
        }

        
    }
}
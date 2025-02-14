The `dstack` server can run anywhere: on your laptop, a dedicated server, or in the cloud.

You can run the server either through `pip` or using Docker.

=== "pip"

    <div class="termy">
    
    ```shell
    $ pip install "dstack[all]" -U
    $ dstack server

    Applying ~/.dstack/server/config.yml...

    The admin token is "bbae0f28-d3dd-4820-bf61-8f4bb40815da"
    The server is running at http://127.0.0.1:3000/
    ```
    
    </div>

    > The server can be set up via `pip` on Linux, macOS, and Windows (via WSL 2).
    > It requires Git and OpenSSH.

=== "Docker"

    <div class="termy">
    
    ```shell
    $ docker run -p 3000:3000 \
        -v $HOME/.dstack/server/:/root/.dstack/server \
        dstackai/dstack

    Applying ~/.dstack/server/config.yml...

    The admin token is "bbae0f28-d3dd-4820-bf61-8f4bb40815da"
    The server is running at http://127.0.0.1:3000/
    ```
        
    </div>

??? info "AWS CloudFormation"
    If you'd like to deploy the server to a private AWS VPC, you can use 
    our CloudFormation [template :material-arrow-top-right-thin:{ .external }](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateURL=https://get-dstack.s3.eu-west-1.amazonaws.com/cloudformation/template.yaml){:target="_blank"}.

    First, ensure, you've set up a private VPC with public and private subnets.

    ![](https://github.com/dstackai/static-assets/blob/main/static-assets/images/dstack-aws-private-vpc-example-v2.png?raw=true)

    Create a stack using the template, and specify the VPC and private subnets.
    Once, the stack is created, go to `Outputs` for the server URL and admin token.

    To access the server URL, ensure you're connected to the VPC, e.g. via VPN client.

    > If you'd like to adjust anything, the source code of the template can be found at
    [`examples/server-deployment/cloudformation/template.yaml` :material-arrow-top-right-thin:{ .external }](https://github.com/dstackai/dstack/blob/master/examples/server-deployment/cloudformation/template.yaml){:target="_blank"}.

## Backend configuration

To use `dstack` with cloud providers, configure [backends](../concepts/backends.md) 
via the `~/.dstack/server/config.yml` file.
The server loads this file on startup. 

Alternatively, you can configure backends on the [project settings page](../guides/administration.md#backends) via UI.

> For using `dstack` with on-prem servers, no backend configuration is required.
> Use [SSH fleets](../concepts/fleets.md#ssh) instead.

## State persistence

By default, the `dstack` server stores its state locally in `~/.dstack/server` using SQLite.

??? info "Replicate state to cloud storage"
    If you’d like, you can configure automatic replication of your SQLite state to cloud object storage using Litestream.

    To enable Litestream replication, set the following environment variables:
    
    - `LITESTREAM_REPLICA_URL` - The url of the cloud object storage.
      Examples: `s3://<bucket-name>/<path>`, `gcs://<bucket-name>/<path>`, `abs://<storage-account>@<container-name>/<path>`, etc.
    
    You also need to configure cloud storage credentials.
    
    **AWS S3**
    
    To persist state into an AWS S3 bucket, provide the following environment variables:
    
    - `AWS_ACCESS_KEY_ID` - The AWS access key ID
    - `AWS_SECRET_ACCESS_KEY` -  The AWS secret access key
    
    **GCP Storage**
    
    To persist state into a GCP Storage bucket, provide one of the following environment variables:
    
    - `GOOGLE_APPLICATION_CREDENTIALS` - The path to the GCP service account key JSON file
    - `GOOGLE_APPLICATION_CREDENTIALS_JSON` - The GCP service account key JSON

    **Azure Blob Storage**
    
    To persist state into an Azure blog storage, provide the following environment variable.
    
    - `LITESTREAM_AZURE_ACCOUNT_KEY` - The Azure storage account key
    
    More [details](https://litestream.io/guides/) on options for configuring replication.

### PostgreSQL

To store the state externally, set the `DSTACK_DATABASE_URL` and `DSTACK_SERVER_CLOUDWATCH_LOG_GROUP` environment variables.

??? info "Migrate from SQLite to PostgreSQL"
    You can migrate the existing state from SQLite to PostgreSQL using `pgloader`:

    1. Create a new PostgreSQL database
    2. Clone the `dstack` repo and [install](https://github.com/dstackai/dstack/blob/master/contributing/DEVELOPMENT.md) `dstack` from source.
       Ensure you've checked out the tag that corresponds to your server version (e.g. `git checkout 0.18.10`).
    3. Apply database migrations to the new database:
      ```bash
      cd src/dstack/_internal/server/
      export DSTACK_DATABASE_URL="postgresql+asyncpg://..."
      alembic upgrade head
      ```
    4. Install [pgloader :material-arrow-top-right-thin:{.external }](https://github.com/dimitri/pgloader){:target="_blank"}
    5. Pass the path to the `~/.dstack/server/data/sqlite.db` file to `SOURCE_PATH` and 
       set `TARGET_PATH` with the URL of the PostgreSQL database. Example:
       ```bash
       cd scripts/
       export SOURCE_PATH=sqlite:///Users/me/.dstack/server/data/sqlite.db
       export TARGET_PATH=postgresql://postgres:postgres@localhost:5432/postgres
       pgloader sqlite_to_psql.load
       ```
       The `pgloader` script will migrate the SQLite data to PostgreSQL. It may emit warnings that are safe to ignore. 
       
       If you encounter errors, please [submit an issue](https://github.com/dstackai/dstack/issues/new/choose).

## Logs storage

By default, `dstack` stores workload logs in `~/.dstack/server/projects/<project_name>/logs`.

### AWS CloudWatch

To store logs in AWS CloudWatch, set the `DSTACK_SERVER_CLOUDWATCH_LOG_GROUP` and
the `DSTACK_SERVER_CLOUDWATCH_LOG_REGION` environment variables. 

The log group must be created beforehand, `dstack` won't try to create it.

??? info "Required permissions"

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
          {
              "Sid": "DstackLogStorageAllow",
              "Effect": "Allow",
              "Action": [
                  "logs:DescribeLogStreams",
                  "logs:CreateLogStream",
                  "logs:GetLogEvents",
                  "logs:PutLogEvents"
              ],
              "Resource": [
                  "arn:aws:logs:::log-group:<group name>",
                  "arn:aws:logs:::log-group:<group name>:*"
              ]
          }
      ]
    }
    ```

## Encryption

By default, `dstack` stores data in plaintext. To enforce encryption, you 
specify one or more encryption keys.

`dstack` currently supports AES and identity (plaintext) encryption keys.
Support for external providers like HashiCorp Vault and AWS KMS is planned.

=== "AES"
    The `aes` encryption key encrypts data using [AES-256](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard) in GCM mode.
    To configure the `aes` encryption, generate a random 32-byte key:

    <div class="termy">
    
    ```shell
    $ head -c 32 /dev/urandom | base64
    
    opmx+r5xGJNVZeErnR0+n+ElF9ajzde37uggELxL
    ```

    </div>
    
    And specify it as `secret`:
    
    ```yaml
    # ...

    encryption:
      keys:
        - type: aes
          name: key1
          secret: opmx+r5xGJNVZeErnR0+n+ElF9ajzde37uggELxL
    ```

=== "Identity"
    The `identity` encryption performs no encryption and stores data in plaintext.
    You can specify an `identity` encryption key explicitly if you want to decrypt the data:

    <div editor-title="~/.dstack/server/config.yml">
    
    ```yaml
    # ...

    encryption:
      keys:
      - type: identity
      - type: aes
        name: key1
        secret: opmx+r5xGJNVZeErnR0+n+ElF9ajzde37uggELxL
    ```

    </div>
    
    With this configuration, the `aes` key will still be used to decrypt the old data,
    but new writes will store the data in plaintext.

??? info "Key rotation"
    If multiple keys are specified, the first is used for encryption, and all are tried for decryption. This enables key
    rotation by specifying a new encryption key.

    <div editor-title="~/.dstack/server/config.yml">
    
    ```yaml
    # ...

    encryption:
      keys:
      - type: aes
        name: key2
        secret: cR2r1JmkPyL6edBQeHKz6ZBjCfS2oWk87Gc2G3wHVoA=

      - type: aes
        name: key1
        secret: E5yzN6V3XvBq/f085ISWFCdgnOGED0kuFaAkASlmmO4=
    ```

    </div>
    
    Old keys may be deleted once all existing records have been updated to re-encrypt sensitive data. 
    Encrypted values are prefixed with key names, allowing DB admins to identify the keys used for encryption.

## Default permissions

By default, all users can create and manage their own projects. You can specify `default_permissions`
to `false` so that only global admins can create and manage projects:

<div editor-title="~/.dstack/server/config.yml">

```yaml
# ...

default_permissions:
  allow_non_admins_create_projects: false
```

</div>

## Backward compatibility

!!! info "Versioning scheme"
    `dstack` follows the `{major}.{minor}.{patch}` versioning scheme based on these principles:

=== "Server"
    The server backward compatibility is maintained across all minor and patch releases. The specific features can be removed but the removal is preceded with deprecation warnings for several minor releases. This means you can use older client versions with newer server versions.

=== "Client"
    The client backward compatibility is maintained across patch releases. A new minor release indicates that the release breaks client backward compatibility. This means you don't need to update the server when you update the client to a new patch release. Still, upgrading a client to a new minor version requires upgrading the server too.

## FAQs

??? info "Can I run multiple replicas of dstack server?"

    Yes, you can if you configure `dstack` to use [PostgreSQL](#postgresql) and [AWS CloudWatch](#aws-cloudwatch).

??? info "Does dstack server support blue-green or rolling deployments?"

    Yes, it does if you configure `dstack` to use [PostgreSQL](#postgresql) and [AWS CloudWatch](#aws-cloudwatch).

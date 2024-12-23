import os
import time
from args import get_args
from ftplib import FTP, error_perm
from loguru import logger


def connect_to_ftp(
    ftp_server, ftp_port, ftp_username, ftp_password, retries=3, timeout=60
):
    attempts = 0
    while attempts < retries:
        try:
            ftp = FTP(ftp_server)
            ftp.port = ftp_port
            ftp.login(user=ftp_username, passwd=ftp_password)
            return ftp
        except Exception as e:
            attempts += 1
            logger.warning(f"Attempt {attempts} failed: {e}")
            if attempts < retries:
                logger.warning(f"Retrying in {timeout} seconds...")
                time.sleep(timeout)
            else:
                logger.warning("All attempts to connect to the FTP server failed.")
                raise


def backup_files_via_ftp(local_folder):
    args = get_args()
    ftp_server = args.ftp_server
    ftp_port = args.ftp_port
    ftp_username = args.ftp_username
    ftp_password = args.ftp_password
    remote_folder = args.ftp_remote_folder

    logger.info("starting audio file backup")

    try:
        ftp = connect_to_ftp(ftp_server, ftp_port, ftp_username, ftp_password)
    except Exception as e:
        print(f"Failed to connect to FTP server: {e}")
        return
    logger.info("successfully connected to FTP server")
    try:
        # Change to the remote folder
        ftp.cwd(remote_folder)

        # List all files in the local folder
        for filename in os.listdir(local_folder):
            local_path = os.path.join(local_folder, filename)

            # Check if it's a file (not a directory)
            if os.path.isfile(local_path):
                with open(local_path, "rb") as file:
                    # Store the file on the FTP server
                    ftp.storbinary(f"STOR {filename}", file)
                # Delete the file after successful upload
                os.remove(local_path)
        logger.info("Audio file backup completed")
    except error_perm as e:
        logger.error(f"FTP error: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Close the connection
        ftp.quit()

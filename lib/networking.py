import os
import ftplib
import gzip

def download_extract_gz_from_ftp(ftp_server, ftp_path, out_path, bs=65536):
    if not os.path.exists(out_path):
        tmp_file = out_path + '.gz'

        # create folder if necessary
        os.makedirs(os.path.dirname(tmp_file), exist_ok=True)

        # download gz if necessary
        if not os.path.exists(tmp_file):
            ftp = ftplib.FTP(ftp_server)
            ftp.login('anonymous', '')
            
            with open(tmp_file, 'wb') as f:
                ftp.retrbinary('RETR %s' % ftp_path, f.write)
        
        # extract gz
        with gzip.open(tmp_file, 'rb') as in_file, \
            open(out_path, 'wb') as out_file:
                while True:
                    block = in_file.read(bs)
                    if not block:
                        break
                    else:
                        out_file.write(block)

        os.remove(tmp_file) 

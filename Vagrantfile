# -*- mode: ruby -*-
# vi: set ft=ruby :

# Francisco Silva - <dpss-helpdesk@inesc-id.pt>

Vagrant.configure("2") do |config|
  # You can search for boxes at https://vagrantcloud.com/search.
  config.vm.box = "generic/ubuntu2204"

  # Forwarded ports example:
  config.vm.network "forwarded_port", guest: 2375, host: 2375
  config.vm.network "forwarded_port", guest: 5000, host: 5000
  config.vm.network "forwarded_port", guest: 6379, host: 6379
  config.vm.network "forwarded_port", guest: 6380, host: 6380

  # Private or public networking examples:
  # config.vm.network "private_network", ip: "192.168.33.10"
  # config.vm.network "public_network"

  # Synced folder example:
  # config.vm.synced_folder "../data", "/vagrant_data"

  config.vm.provider "virtualbox" do |vb|
    vb.name = "ubuntu-24.04-vm"
    vb.gui = false
    vb.memory = "30720"
    vb.cpus = 4
    vb.customize ["modifyvm", :id, "--nested-hw-virt", "on"]
  end
  
  config.vm.synced_folder "./", "/octoflows"

  # Provisioning: install common packages
  config.vm.provision "shell", inline: <<-SHELL
    sudo apt-get update
    sudo apt-get install -y curl unzip
    # Mine
    sudo apt-get install -y graphviz redis-tools
    sudo apt-get install -y python3-pip python3-venv

    # Download and run the official Docker install script
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    sudo usermod -aG docker vagrant

    # Clone project and install Python dependencies
    ls /octoflows
    pip install -r /octoflows/src/requirements.txt

    echo "--- Configuring Docker Remote API (NEW) ---"
    mkdir -p /etc/systemd/system/docker.service.d

    cat > /etc/systemd/system/docker.service.d/override.conf <<EOF

    # 3. Reload systemd and restart Docker to apply changes
    systemctl daemon-reload
    systemctl restart docker
    
    sudo apt-get clean
  SHELL
end
